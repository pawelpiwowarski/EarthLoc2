import csv
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import parser  # Reusing your existing parser
import commons


def create_path_mapping(args):
    """
    Main function to generate a FAISS index to local file path mapping CSV.
    """
    # 1. SETUP
    start_time = datetime.now()
    args.log_dir = Path("logs/create_mapping") / start_time.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.log_dir, console="info")
    logging.info("Starting FAISS index to local path mapping creation.")

    # 2. GET DATABASE IMAGE PATHS
    # This order MUST match the order used to create the FAISS index.
    db_root = args.dataset_path / "database"
    logging.info(f"Scanning for database images in {db_root}...")
    all_paths = sorted(list(db_root.glob("*/*/*.jpg")))

    logging.info("Filtering paths to include only years >= 2021...")
    db_image_paths = []
    for path in all_paths:
        try:
            # Directory name is expected to be in 'YEAR_ZOOM' format
            year_str = path.parent.parent.name.split("_")[0]
            if int(year_str) >= 2021:
                db_image_paths.append(path)
        except (ValueError, IndexError):
            logging.debug(f"Skipping path with unexpected dir format: {path}")
            continue

    num_db_images = len(db_image_paths)

    if num_db_images == 0:
        logging.error(
            f"No images found in {db_root} for years >= 2021. Check --dataset_path."
        )
        sys.exit(1)

    logging.info(
        f"Found {num_db_images} database images from 2021 onwards. "
        f"This will result in {num_db_images * 4} FAISS indices."
    )

    # 3. CREATE THE MAPPING FILE
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["faiss_index", "local_path"])

        for i, image_path in enumerate(
            tqdm(db_image_paths, desc="Generating path mapping")
        ):
            # The local path is simply the string representation of the Path object
            local_path_str = str(image_path)

            # Write the 4 mappings for this image (0, 90, 180, 270 deg)
            # All four rotated feature vectors in the FAISS index map back
            # to the same original image file.
            writer.writerow([i, local_path_str])
            writer.writerow([i + num_db_images, local_path_str])
            writer.writerow([i + 2 * num_db_images, local_path_str])
            writer.writerow([i + 3 * num_db_images, local_path_str])

    logging.info(f"Successfully created mapping file at: {output_csv_path}")


if __name__ == "__main__":

    args = parser.parse_arguments()
    create_path_mapping(args)