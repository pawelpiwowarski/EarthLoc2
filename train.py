import os
import sys
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import torch
import einops
import logging
import torchmetrics
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from loss import DynamicallyNeutralMinerLoss, OrdinalTripletLossCosine
from miner import OrdinalQuadrupletMiner
import eval
import test
import parser
import commons
import augmentations
import compute_clusters
from apl_models.apl_model import APLModel
from apl_models.apl_model_dinov2 import DINOv2FeatureExtractor
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
import query_mining


args = parser.parse_arguments()
start_time = datetime.now()
args.log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
commons.make_deterministic(args.seed)
commons.setup_logging(args.log_dir, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.log_dir}")
secondary_model = None
if args.model == "VGG":
	model = APLModel()

elif args.model == "Dinov2":
	if args.model_type == "small":
		model_type = "vit_small_patch14_reg4_dinov2.lvd142m"
		desc_dim = 384
	elif args.model_type == "base":
		model_type = "vit_base_patch14_reg4_dinov2.lvd142m"
		desc_dim = 768
	elif args.model_type == "large":
		model_type = "vit_large_patch14_reg4_dinov2.lvd142m"
		desc_dim = 1024

	model = DINOv2FeatureExtractor(
		model_type=model_type,
		num_of_layers_to_unfreeze=args.num_of_layers_to_unfreeze,
		desc_dim=desc_dim,
		aggregator_type=args.aggregator_type,
	)


else:
	raise NotImplementedError("The model is not yet supported")


# Make the model run fast if using Ampere or Hopper chips
if torch.cuda.is_available():
	props = torch.cuda.get_device_properties(torch.cuda.current_device())
	if props.major >= 8:
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		logging.info(f"TF32 enabled on GPU {props.name}")
	else:
		logging.info(f"TF32 not supported on GPU {props.name}")

#### LOSSES & OPTIM ####
criterion = losses.MultiSimilarityLoss(
	alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
)
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

## starting with 0 values for the default arguments
start_epoch = best_r5 = not_improved_num = 0
clustered_paths = None  # Initialize clustered_paths to be loaded from checkpoint

# Handle resume from checkpoint
if args.resume_model:
	logging.info(f"Resuming training from checkpoint {args.resume_model}")
	# Load checkpoint to CPU first to avoid GPU memory issues
	ckpt = torch.load(args.resume_model, map_location="cpu")

	# Restore model + optimizer + scaler
	model.load_state_dict(ckpt["model_state_dict"])
	optim.load_state_dict(ckpt["optimizer_state_dict"])
	if "scaler_state_dict" in ckpt:
		scaler.load_state_dict(ckpt["scaler_state_dict"])

	# Manually move optimizer state to the correct device
	for state in optim.state.values():
		for k, v in state.items():
			if isinstance(v, torch.Tensor):
				state[k] = v.to(args.device)

	# Restore epoch counters / best metrics / cluster paths
	start_epoch = ckpt.get("epoch_num", 0)
	best_r5 = ckpt.get("best_r5", 0.0)
	not_improved_num = ckpt.get("not_improved_num", 0)
	clustered_paths = ckpt.get("clustered_paths")  # Load clustered paths

	logging.info(
		f"Resumed at epoch {start_epoch}, best_r5={best_r5:.3f}, "
		f"patience={not_improved_num}/{args.patience}"
	)
	if clustered_paths is not None:
		logging.info("Successfully resumed clustered_paths from checkpoint.")
	else:
		logging.warning(
			"Could not find clustered_paths in checkpoint, will start with random clusters."
		)

#### DATA ####
db_paths = list((args.dataset_path / "database").glob("*/*/*.jpg"))

# If not resuming, or if checkpoint is old, create initial random clusters
if clustered_paths is None:
	logging.info("Creating initial random clusters for training.")
	clustered_paths = TrainDataset.create_random_clusters(
		db_paths, args.num_clusters
	)

train_dataset = TrainDataset(
	clustered_paths=clustered_paths,
	batch_size=args.batch_size,
	size_before_transf=args.size_before_transf,
)

val_dataset = TestDataset(
	dataset_path=args.dataset_path,
	dataset_name="val",
	db_paths=db_paths,
	image_size=model.image_size,
	# Queries near Texas used for validation
	# We use more queries and more db images as with the basic settings
	# the recall@5 of the best models would quickly reach 99%
	center_lat=30,
	center_lon=-95,
	thresh_queries=1000,
	thresh_db=2200,
)

dataloader = torch.utils.data.DataLoader(
	train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True
)

augmentation = augmentations.get_my_augment(
	distortion_scale=args.dist_scale,
	crop_size=args.crop_size,
	final_size=model.image_size,
	rand_rot=args.rand_rot,
	brightness=args.brightness,
	contrast=args.contrast,
	saturation=args.saturation,
	hue=args.hue,
)
query_augmentation = augmentations.get_query_augment(
	rand_rot=args.rand_rot, final_size=model.image_size
)

model = model.to(args.device)

for num_epoch in range(start_epoch, args.num_epochs):
	if num_epoch != 0 and num_epoch % args.compute_clusters_every_n_epochs == 0:
	
		clustered_paths = compute_clusters.compute_clusters(
			model,
			all_paths=db_paths,
			num_clusters=args.num_clusters,
			num_workers=args.num_workers,
		)

		

		train_dataset = TrainDataset(
			clustered_paths=clustered_paths,
			batch_size=args.batch_size,
			size_before_transf=args.size_before_transf,
		)
		dataloader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=1,
			num_workers=args.num_workers,
			drop_last=True,
			shuffle=True,
		)

	model = model.train()
	mean_loss = torchmetrics.MeanMetric()
	mean_batch_acc = torchmetrics.MeanMetric()
	tqdm_bar = tqdm(dataloader, total=args.iterations_per_epoch, ncols=120)
	for iteration, (images, is_overlapping, chosen_paths) in enumerate(tqdm_bar):
		if iteration >= args.iterations_per_epoch:
			break
		with torch.cuda.amp.autocast():
			images = einops.rearrange(
				images,
				"one bs years c h w -> (one bs) years c h w",
				bs=args.batch_size,
				one=1,
				years=4,
			)
			images = images.to(args.device)

			# Apply same augmentation to images from same year, i.e. Year-Wise Augmentation
			views = [augmentation(images[:, year]) for year in range(4)]

			views = einops.rearrange(
				views, "nv b c h w -> (b nv) c h w", nv=4, b=args.batch_size
			)

			labels = torch.repeat_interleave(
				torch.arange(args.batch_size), 4
			).to(args.device)

			output_plots = num_epoch == 0 and iteration > 0 and iteration < 25

			query_images, query_labels = query_mining.get_query_candidates(
				chosen_paths, iou=args.iou_thresh, output_plots=output_plots
			)

			# if we are mining queries and the query labels exists then add them to the current batch
			if args.mine_queries and query_labels is not None:
				query_labels = torch.tensor(query_labels).to(args.device)

				if query_labels.dim() > 0:
					query_labels = query_labels.squeeze(1)
				query_images = query_augmentation(query_images).to(args.device)

				if output_plots:
					# De-normalization tensors
					mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
					std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

					def denorm(img):
						return (img * std + mean).clamp(0, 1)

					# Find the batch-index of this label in the candidate views
					candidate_index = query_labels[0].item()
					mask = labels == candidate_index
					indices = mask.nonzero(as_tuple=True)[0]

					# Grab the four candidate images (one per year)
					cand_images = []
					for idx in indices:
						cand_images.append(denorm(views[idx].cpu().detach()))

					# Grab all mined query images for that label
					mask = (
						query_labels == candidate_index
					)  # shape (N,), dtype=bool

					# 2) select only those query images
					query_imgs = [
						denorm(img.cpu().detach()) for img in query_images[mask]
					]

					# Build titles
					num_years = len(cand_images)
					num_queries = len(query_imgs)
					titles = [f"Candidate {2018 + i}" for i in range(num_years)]
					titles += [f"Query {i+1}" for i in range(num_queries)]

					# Make a wide figure
					total_cols = num_years + num_queries
					fig, axes = plt.subplots(
						1,
						total_cols,
						figsize=(4 * total_cols, 4),
						constrained_layout=False,
					)
					axes = np.atleast_1d(axes).flatten()

					# Make room at the top for titles
					fig.subplots_adjust(
						top=0.80
					)  # lower this value to give more space

					# Plot each image and bump its title down by pad=15 points
					for ax, img, title in zip(
						axes, cand_images + query_imgs, titles
					):
						np_img = img.permute(1, 2, 0).numpy()
						ax.imshow(np_img)
						ax.axis("off")
						ax.set_title(title, fontsize=12, pad=15)

					# Save the figure
					plt.savefig(
						f"./plots/augmented_example_{iteration}.png",
						dpi=150,
						bbox_inches="tight",
						pad_inches=0.1,
					)
					plt.close()

				views = torch.cat([views, query_images])
				descriptors = model(views)
				labels = torch.cat([labels, query_labels])

				miner_outputs = miner(descriptors, labels)
				# Filter away overlapping pairs of images, i.e. Neutral-Aware MS loss
				N = (
					views.shape[0] - query_images.shape[0]
				)  # Number of original images
				# or, if you have the original batch size stored, use that

				# Only keep negatives where both anchor and negative are from the original batch
				all_negative_anchors, negative_pairs = miner_outputs[2:]
				mask = (all_negative_anchors < N) & (negative_pairs < N)

				anchors_filtered_negatives = all_negative_anchors[mask]
				negatives_filtered = negative_pairs[mask]

				is_non_overlapping = (
					is_overlapping.to(args.device)[
						0,
						anchors_filtered_negatives // 4,
						negatives_filtered // 4,
					]
					== 0
				)
				far_indexes = torch.where(is_non_overlapping)[0]
				anchors_filtered_negatives = anchors_filtered_negatives[
					far_indexes
				]
				negatives_filtered = negatives_filtered[far_indexes]
				miner_outputs_filtered = (
					miner_outputs[0],
					miner_outputs[1],
					anchors_filtered_negatives,
					negatives_filtered,
				)
			# if we are not mining queries or they do not exists hen proceed with a standard setup
			else:
				descriptors = model(views)
				miner_outputs = miner(descriptors, labels)
				# Filter away overlapping pairs of images, i.e. Neutral-Aware MS loss
				all_negative_anchors, negative_pairs = miner_outputs[2:]

				is_non_overlapping = (
					is_overlapping.to(args.device)[
						0, all_negative_anchors // 4, negative_pairs // 4
					]
					== 0
				)
				far_indexes = torch.where(is_non_overlapping)[0]
				# only true negatives anchors
				anchors_filtered_negatives = all_negative_anchors[far_indexes]
				# only true negative pairs
				negatives_filtered = negative_pairs[far_indexes]
				# miner output filtered without the neutral cases
				miner_outputs_filtered = tuple(
					[
						miner_outputs[0],
						miner_outputs[1],
						anchors_filtered_negatives,
						negatives_filtered,
					]
				)

			loss = criterion(descriptors, labels, miner_outputs_filtered)

		scaler.scale(loss).backward()

		# calculate the % of trivial pairs/triplets which do not contribute in the loss value
		nb_samples = descriptors.shape[0]
		nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
		batch_acc = (1.0 - (nb_mined / nb_samples)) * 100

		scaler.step(optim)
		scaler.update()
		optim.zero_grad()
		mean_loss.update(loss.item())
		mean_batch_acc.update(batch_acc)
		tqdm_bar.desc = (
			f"Loss: {mean_loss.compute()} - batch_acc: {batch_acc:.1f} - "
			f"{nb_samples} - {nb_mined}"
		)

	# … after your training loop for this epoch …
	recalls, recalls_str = test.test(val_dataset, model, device=args.device)
	r5 = recalls[1]
	logging.debug(f"Recalls: {recalls_str}")

	# ----------------------------------------------------------------
	# If early stopping is disabled, always overwrite the single ckpt
	# ----------------------------------------------------------------
	is_best = r5 > best_r5

	if not args.early_stopping:
		# delete any existing ckpt_*.torch files
		for best in args.log_dir.glob("best_model_*.torch"):
			best.unlink()
		# save new best
		torch.save(
			model.state_dict(), args.log_dir / f"best_model_{r5:.1f}.torch"
		)

		# remove any old ckpt_*.torch
		for ckpt_path in args.log_dir.glob("ckpt_*.torch"):
			ckpt_path.unlink()
		# save checkpoint for resume
		torch.save(
			{
				"epoch_num": num_epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optim.state_dict(),
				"scaler_state_dict": scaler.state_dict(),
				"best_r5": best_r5,
				"not_improved_num": not_improved_num,
				"clustered_paths": clustered_paths,
			},
			args.log_dir / f"ckpt_e{num_epoch:02d}_{r5:.1f}.torch",
		)
		logging.info(
			f"Epoch {num_epoch:>2} – loss: {mean_loss.compute():.2f} – "
			f"batch_acc: {mean_batch_acc.compute():.1f} – "
		)
		# skip the rest of the early‐stopping logic
		continue

	if is_best:
		# remove old best_model_*.torch
		for best in args.log_dir.glob("best_model_*.torch"):
			best.unlink()
		# save new best
		torch.save(
			model.state_dict(), args.log_dir / f"best_model_{r5:.1f}.torch"
		)

		# remove any old ckpt_*.torch
		for ckpt_path in args.log_dir.glob("ckpt_*.torch"):
			ckpt_path.unlink()
		# save checkpoint for resume
		torch.save(
			{
				"epoch_num": num_epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optim.state_dict(),
				"scaler_state_dict": scaler.state_dict(),
				"best_r5": r5,  # Save the new best r5
				"not_improved_num": 0,
				"clustered_paths": clustered_paths,
			},
			args.log_dir / f"ckpt_e{num_epoch:02d}_{r5:.1f}.torch",
		)

		logging.debug(
			f"Improved: previous best r5 = {best_r5:.1f}, current r5 = {r5:.1f}"
		)
		best_r5 = r5
		not_improved_num = 0
	else:
		not_improved_num += 1
		logging.debug(
			f"Not improved: {not_improved_num}/{args.patience} – "
			f"best r5 = {best_r5:.1f}, current r5 = {r5:.1f}"
		)

	logging.info(
		f"Epoch {num_epoch:>2} – loss: {mean_loss.compute():.2f} – "
		f"batch_acc: {mean_batch_acc.compute():.1f} – "
		f"patience left: {args.patience - not_improved_num} – "
		f"best r5: {best_r5:.1f} – {recalls_str[:20]}"
	)

	if not_improved_num >= args.patience:
		logging.info(
			f"Performance did not improve for {not_improved_num} epochs. "
			"Stopping training."
		)
		break

logging.info(f"Training finished in {str(datetime.now() - start_time)[:-7]}")

logging.debug("Testing with the best model")

best_model_path = list(args.log_dir.glob("best_*"))[0]
best_model_state_dict = torch.load(best_model_path)
model.load_state_dict(best_model_state_dict)

eval.eval_on_all_test_sets(
	model,
	args.dataset_path,
	db_paths,
	args.log_dir,
	num_preds_to_save=args.num_preds_to_save,
	device=args.device,
)