import torch
import random
import logging
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from torchvision.ops import box_iou
import datasets.utils as utils




class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, clustered_paths, batch_size=32, size_before_transf=800):
        self.batch_size = batch_size
        self.size_before_transf = size_before_transf

        # Filter paths to only include European images and flatten the structure
        self.clustered_paths = clustered_paths

        
        if not self.clustered_paths:
            raise ValueError("No images found in the dataset")
            
        num_all_paths = sum(len(cluster) for cluster in self.clustered_paths)
        logging.debug(f"TrainDataset has {num_all_paths} * 4 images across {len(self.clustered_paths)} clusters")
    
    def __getitem__(self, index):
        # Select a random cluster that has enough images
        while True:
            cluster = random.choice(self.clustered_paths)
            if len(cluster) >= self.batch_size or len(cluster) > 0:
                break

        def are_tiles_overlapping(path1, path2):
            min_lat1, min_lon1, _, _, max_lat1, max_lon1, _, _, = utils.get_footprint_from_path(path1)
            min_lat2, min_lon2, _, _, max_lat2, max_lon2, _, _, = utils.get_footprint_from_path(path2)
            if max_lat1 <= min_lat2 or max_lat2 <= min_lat1: return False
            if max_lon1 <= min_lon2 or max_lon2 <= min_lon1: return False
            return True
        
        # Sample paths without replacement if possible, with replacement if necessary
        replace = len(cluster) < self.batch_size
        chosen_paths = np.random.choice(cluster, size=self.batch_size, replace=replace).tolist()
        
        # Load and preprocess images
        images = torch.zeros([self.batch_size, 4, 3, self.size_before_transf, self.size_before_transf], 
                           dtype=torch.float32)
        
        for i, path in enumerate(chosen_paths):
            year_paths = [utils.replace_year_in_path(path, 2021, y) for y in range(2018, 2022)]
            
            for j, year_path in enumerate(year_paths):
                try:
                    img = Image.open(year_path)
                    img = tfm.Resize(self.size_before_transf)(img)
                    images[i, j] = tfm.ToTensor()(img)
                except Exception as e:
                    logging.warning(f"Error loading {year_path}: {str(e)}")
                    # Fill with random noise if image fails to load
                    images[i, j] = torch.rand(3, self.size_before_transf, self.size_before_transf)

        # Compute overlapping matrix
  
        is_overlapping = np.zeros([len(chosen_paths), len(chosen_paths)], dtype=np.int8)
        for i1, p1 in enumerate(chosen_paths):
            for i2, p2 in enumerate(chosen_paths):
                if are_tiles_overlapping(p1, p2):
                    is_overlapping[i1, i2] = 1
        
        
        # Convert paths to strings
        chosen_paths = [str(p) for p in chosen_paths]
        
        return images, is_overlapping, chosen_paths
    
    def __len__(self):
        return 100_000  # Large number for epoch-based training
    
    @staticmethod
    def create_random_clusters(paths, num_clusters):
        tmp_list = paths.copy()
        random.shuffle(tmp_list)
        clustered_paths = np.array_split(tmp_list, num_clusters)
        return clustered_paths