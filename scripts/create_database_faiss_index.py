import sys
import torch
import logging
import faiss
import einops
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from PIL import Image

import parser
import commons
from apl_models.apl_model import APLModel
from apl_models.apl_model_dinov2 import DINOv2FeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfm


class DatabaseDataset(Dataset):
    """
    A simple dataset to load database images for feature extraction.
    It returns the image tensor and its original index.
    """

    def __init__(self, db_paths, image_size):
        super().__init__()
        self.db_paths = db_paths
        self.transform = tfm.Compose(
            [
                tfm.Resize(image_size),
                tfm.ToTensor(),
                tfm.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.db_paths)

    def __getitem__(self, index):
        image_path = self.db_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, index


def create_index(args):
    """
    Main function to create and save a FAISS index for the database images.
    """
    # 1. SETUP
    start_time = datetime.now()
    args.log_dir = Path("logs") / "create_index" / start_time.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.log_dir, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.log_dir}")

    # 2. MODEL INITIALIZATION
    if args.model == "VGG":
        model = APLModel()
        model_name_str = f"{args.model}"
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
        else:
            raise ValueError(f"Unknown Dinov2 model type: {args.model_type}")

        model = DINOv2FeatureExtractor(
            model_type=model_type,
            num_of_layers_to_unfreeze=args.num_of_layers_to_unfreeze,
            desc_dim=desc_dim,
            aggregator_type=args.aggregator_type,
        )
        model_name_str = (
            f"{args.model}_{args.model_type}_{args.aggregator_type}"
        )

    model = model.to(args.device)

    if args.resume_model is None:
        logging.error(
            "FATAL: A model checkpoint must be provided via --resume_model"
        )
        sys.exit(1)

    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
    model = model.eval()

    # 3. DATA PREPARATION (with filtering for year >= 2021)
    db_root = args.dataset_path / "database"
    logging.info(f"Scanning for all database images in {db_root}...")
    all_paths = sorted(list(db_root.glob("*/*/*.jpg")))

    logging.info(f"Found {len(all_paths)} total images. Filtering for years >= 2021...")
    db_paths = []
    for path in all_paths:
        try:
            # Directory name is expected to be in 'YEAR_ZOOM' format
            year_str = path.parent.parent.name.split("_")[0]
            if int(year_str) >= 2021:
                db_paths.append(path)
        except (ValueError, IndexError):
            logging.debug(f"Skipping path with unexpected dir format: {path}")
            continue

    if not db_paths:
        logging.error(f"No images found in {db_root} for years >= 2021. Aborting.")
        sys.exit(1)

    logging.info(f"Processing {len(db_paths)} images from 2021 onwards.")

    db_dataset = DatabaseDataset(db_paths, model.image_size)
    dataloader = DataLoader(
        dataset=db_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 4. FEATURE EXTRACTION
    logging.info("Extracting features from all database images...")
    all_descriptors = np.empty(
        (4, len(db_dataset), model.desc_dim), dtype="float32"
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, indices in tqdm(dataloader, desc="Extracting"):
                for rot_idx, angle in enumerate([0, 90, 180, 270]):
                    images_gpu = images.to(args.device)
                    rot_images = tfm.functional.rotate(images_gpu, angle)
                    descriptors = model(rot_images)
                    all_descriptors[rot_idx, indices.numpy(), :] = (
                        descriptors.cpu().numpy()
                    )

    # Reshape descriptors to be (num_images * 4, desc_dim)
    db_descriptors = einops.rearrange(
        all_descriptors, "four db dim -> (four db) dim"
    )
    logging.info(
        f"Extracted {db_descriptors.shape[0]} descriptors of dimension {db_descriptors.shape[1]}"
    )

    # 5. FAISS INDEXING
    logging.info("Building FAISS index...")
    faiss_index = faiss.IndexFlatL2(model.desc_dim)
    faiss_index.add(db_descriptors)
    logging.info(f"FAISS index created. Total vectors in index: {faiss_index.ntotal}")

    # 6. SAVING THE INDEX (with the corrected path)
    # This will now create the directory relative to your project folder
    output_dir = Path(f"faiss/{model_name_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index_2021.bin"

    logging.info(f"Saving index to {index_path}")
    faiss.write_index(faiss_index, str(index_path))
    logging.info("Index saved successfully.")


if __name__ == "__main__":
    # Ensure your parser includes --batch_size and --num_workers
    args = parser.parse_arguments()
    create_index(args)