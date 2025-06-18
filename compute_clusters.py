
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm

import datasets.utils as utils
from datasets.base_dataset import BaseDataset


def compute_clusters(model, all_paths, num_clusters=100,
                     device="cuda", num_workers=8, batch_size=32):
    
    # Keep only DB images from 2021
    all_paths = [p for p in all_paths if utils.get_year_from_path(p) == 2021]
    test_dataset = BaseDataset(all_paths, model.image_size)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, num_workers=num_workers, batch_size=batch_size
    )
    model = model.eval()
    all_descs = np.empty((len(test_dataset), model.desc_dim), dtype=np.float32)
    
    with torch.inference_mode():
        for images, indices, _ in tqdm(dataloader, ncols=120,
                                    desc="Computing descriptors for clustering"):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descs[indices.numpy(), :] = descriptors
    
    logging.debug(f"Start computing clusters with {len(all_paths)} paths")
    
    
    kmeans = faiss.Kmeans(model.desc_dim, num_clusters, niter=100, verbose=True)
    kmeans.train(all_descs)
    cluster_ids_x = kmeans.index.search(all_descs, 1)[1][:, 0]
    
    clustered_paths = [[] for _ in range(num_clusters)]
    for cl_id, path in zip(cluster_ids_x, all_paths):
        clustered_paths[int(cl_id)].append(path)
    
    return clustered_paths


def compute_cluster_geographic(all_paths, num_clusters=100, num_workers=8):
    """
    Cluster image paths geographically based on the center of their
    footprint.

    This function filters the paths (here using year 2021 as in the original
    example), extracts the geographic footprint from each path, computes the
    center (average latitude and longitude), and clusters these centers into
    `num_clusters` clusters using Faiss K-means.

    Args:
        all_paths (List[str]): List of paths.
        num_clusters (int): Number of clusters to form.
        num_workers (int): Number of workers to use (reserved for possible
                           parallelization improvements).

    Returns:
        List[List[str]]: A list of clusters with each cluster containing the
                         image paths whose geographic center falls into the
                         corresponding cluster.
    """
    # Filter paths by year if desired (using 2021 as an example)
    all_paths = [p for p in all_paths if utils.get_year_from_path(p) == 2021]
    num_paths = len(all_paths)
    if num_paths == 0:
        logging.warning("No paths found for the specified filter criteria.")
        return []

    # Prepare an array to store geographic centers.
    centers = np.empty((num_paths, 2), dtype=np.float32)
    
    # Compute the geographic center (average latitude and longitude)
    for idx, path in tqdm(
        enumerate(all_paths),
        total=num_paths,
        ncols=120,
        desc="Computing geographic centers",
    ):
        # Extract the four corner coordinates (lat, lon pairs)
        lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4 = (
            utils.get_footprint_from_path(path)
        )
        center_lat = (lat1 + lat2 + lat3 + lat4) / 4.0
        center_lon = (lon1 + lon2 + lon3 + lon4) / 4.0
        centers[idx, 0] = center_lat
        centers[idx, 1] = center_lon

    logging.debug("Starting geographic clustering on computed centers.")

    # Use Faiss K-means to cluster the centers.
    # Note: The cluster dimension is 2 (lat and lon).
    kmeans = faiss.Kmeans(2, num_clusters, niter=100, verbose=True)
    kmeans.train(centers)
    # The search returns distances and cluster ids; we only need the ids.
    cluster_ids = kmeans.index.search(centers, 1)[1][:, 0]

    # Aggregate paths into clusters.
    clustered_paths = [[] for _ in range(num_clusters)]
    for cl_id, path in zip(cluster_ids, all_paths):
        clustered_paths[int(cl_id)].append(path)

    return clustered_paths

def compute_hybrid_clusters(model,
                            all_paths,
                            geo_clusters=10,
                            embed_clusters=5,
                            device="cuda",
                            num_workers=8,
                            batch_size=32):
    """
    Hybrid clustering: first groups image paths into geographic clusters,
    and then subclusters each geographic cluster using the image embeddings.

    Args:
        model: The model to compute image embeddings.
        all_paths (List[str]): List of image paths.
        geo_clusters (int): Number of clusters for geographic clustering.
        embed_clusters (int): Number of clusters for embedding-based subclustering.
        device (str): Device on which to run the model.
        num_workers (int): Number of workers for the dataloader.
        batch_size (int): Batch size for embedding computation.

    Returns:
        List[List[str]]: A list of clusters with each cluster containing image paths.
    """
    # Step 1: Geographic clustering
    geo_clustered_paths = compute_cluster_geographic(
        all_paths, num_clusters=geo_clusters, num_workers=num_workers
    )

    final_clusters = []

    # Process each geographic cluster individually.
    for i, cluster_paths in enumerate(geo_clustered_paths):
        if len(cluster_paths) == 0:
            continue

        # Create a dataset and dataloader for the current geographic cluster.
        dataset = BaseDataset(cluster_paths, model.image_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False
        )

        # Allocate space for embeddings
        num_images = len(dataset)
        embeddings = np.empty((num_images, model.desc_dim), dtype=np.float32)

        # Compute embeddings for all images in the geographic cluster
        model.eval()
        with torch.inference_mode():
            for images, indices, _ in tqdm(
                dataloader,
                ncols=120,
                desc=f"Computing embeddings for geo cluster {i}",
            ):
                images = images.to(device)
                descriptors = model(images)
                descriptors = descriptors.cpu().numpy()
                embeddings[indices.numpy(), :] = descriptors

        # Step 2: Subcluster using embeddings within this geographic cluster.
        logging.debug(
            f"Starting embedding-based clustering for geographic cluster {i}"
        )
        kmeans_emb = faiss.Kmeans(
            model.desc_dim, embed_clusters, niter=100, verbose=True
        )
        kmeans_emb.train(embeddings)
        cluster_ids_emb = kmeans_emb.index.search(embeddings, 1)[1][:, 0]

        # Group the paths by their embedding cluster id.
        geo_subclusters = [[] for _ in range(embed_clusters)]
        for cid, path in zip(cluster_ids_emb, cluster_paths):
            geo_subclusters[int(cid)].append(path)

        # Append all subclusters of this geographic cluster to the final clusters.
        final_clusters.extend(geo_subclusters)

    return final_clusters