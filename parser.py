import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=96, help="_")
    parser.add_argument("--patience", type=int, default=10, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=500, help="_")
    parser.add_argument("--num_epochs", type=int, default=20, help="_")
    parser.add_argument("--lr", type=float, default=0.0001, help="_")

    parser.add_argument("--compute_clusters_every_n_epochs", type=int, default=4, help="_")
    parser.add_argument("--num_clusters", type=int, default=50, help="_")
    
    # Data augmentation
    parser.add_argument("--size_before_transf", type=int, default=800,
                        help="image size before applying augmentations")
    parser.add_argument("--crop_size", type=int, default=700,
                        help="size of random crop")
    parser.add_argument("--image_size", type=int, default=320,
                        help="image size for train and test")
    parser.add_argument("--rand_rot", type=int, default=45,
                        help="random rotation augmentation")
    parser.add_argument("--dist_scale", type=float, default=0.5,
                        help="distortion augmentation")
    parser.add_argument("--brightness", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--contrast", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--saturation", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--hue", type=float, default=0.0,
                        help="color jittering")
    
    # Others
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=3, help="_")
    
    parser.add_argument("--resume_model", type=str, default=None,
                        help="pass the path of a best_model.torch file to load its weights")
    
    # Visualizations
    parser.add_argument("--num_preds_to_save", type=int, default=20,
                        help="Save visualizations of N queries and their predictions")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default="./data", help="_")
    parser.add_argument("--log_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/log_dir")

    #Extras added by me

    parser.add_argument('--model', type=str, default = "VGG")
    parser.add_argument('--model_type',type=str,default="base", help='Which type of DinoV2 model to use, small, base, large')
    parser.add_argument('--num_of_layers_to_unfreeze',default=1, type=int, help='How many layers to freeze when using the DinoV2 model ')
    parser.add_argument('--aggregator_type', type=str, default='No', help='Which aggregator to use after the feature extractor, can be SALAD or Mixvpr' )
    parser.add_argument('--mine_queries', action='store_true',default=False, help='Whether to mine the queries too alongside the candidates')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='The default threshold used when mining the queries ')
    parser.add_argument('--evaluate_on_full_db', default=False,action='store_true', help='Whether to evaluate using the full databse in contrast to a part of it used in computing the test recalls' )
    parser.add_argument('--early_stopping', default=False,action='store_true', help='Whether to use early stopping i.e have a patience on the validation recall then use only the ' )
    
    
  
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./faiss/faiss_index_to_url.csv",
        help="Path to save the output CSV file.",
    )
    
    
    args = parser.parse_args()
    args.dataset_path = Path(args.dataset_path)
    
    return args



