
import os
import sys
import torch
import random
import logging
import traceback
import numpy as np


def make_deterministic(seed: int = 0):
    """Make results deterministic. If seed == -1, do not make deterministic.
        Running your script in a deterministic way might slow it down.
        Note that for some packages (eg: sklearn's PCA) this function is not enough.
    """
    seed = int(seed)
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_folder: str, exist_ok: bool = False, console: str = "debug",
                  info_filename: str = "info.log", debug_filename: str = "debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        exist_ok (boolean): if False throw a FileExistsError if output_folder already exists
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if not exist_ok and os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    
    if info_filename is not None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    def my_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
        logging.info("Experiment finished (with some errors)")
    sys.excepthook = my_handler


def find_connected_components(positive_connections):
    """
    Compute connected components for a batch of images based on IoU > 0.5.
    Returns a tensor of component labels for each original image.
    """
    batch_size = positive_connections.shape[0]
    parent = torch.arange(batch_size, device=positive_connections.device)

    # Union-Find to find connected components
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            if root_u < root_v:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v

    # Union all connected pairs
    for i in range(batch_size):
        for j in range(batch_size):
            if positive_connections[i, j]:
                union(i, j)

    # Assign component labels (smallest index in the component)
    component_labels = torch.tensor([find(i) for i in range(batch_size)], device=positive_connections.device)
    return component_labels
