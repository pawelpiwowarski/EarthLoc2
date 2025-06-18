import os
import random
import re
import logging
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as tfm
import matplotlib.pyplot as plt

def get_incremented_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def add_queries_prefix(path):
    prefix = "data/queries/"
    return path if path.startswith(prefix) else prefix + path

def clean_path(path):
    if path.startswith('data/'):
        path = path[len('data/'):]
    return re.sub(r'(2018|2019|2020)', '2021', path)

def get_query_candidates(
    candidate_paths,
    iou=0.5,
    size_before_transf=800,
    output_plots=True
):
    """
    For each candidate, pick at most one query whose IOU ≥ threshold.
    Returns:
      images:  Tensor [N,3,H,W]
      labels:  list of [candidate_idx] of length N
    """
    # load precomputed candidate→queries map
    iou_str = f"{iou:.2f}"
    idx_path = f"./data/candidate_to_queries_with_db_2021_iou_{iou_str}.pt"
    candidate_to_queries = torch.load(idx_path)

    # normalize candidate paths for lookup
    cand_strs = [clean_path(str(Path(p[0]))) for p in candidate_paths]
    seen_queries = set()
    query_paths  = []
    query_labels = []

    # one query max per candidate
    for cand_idx, cand in enumerate(cand_strs):
        all_qs = candidate_to_queries.get(cand, [])
        # shuffle so we don't always pick the same one first
        random.shuffle(all_qs)
        # pick the first unseen query
        chosen = None
        for q in all_qs:
            if q not in seen_queries:
                chosen = q
                break
        if chosen is None:
            continue
        seen_queries.add(chosen)
        query_paths.append(add_queries_prefix(chosen))
        query_labels.append([cand_idx])

    if not query_paths:
        return None, None

    # load images into a tensor
    images = torch.zeros(
        (len(query_paths), 3, size_before_transf, size_before_transf),
        dtype=torch.float32
    )
    for i, qp in enumerate(query_paths):
        try:
            img = Image.open(qp)
            img = tfm.Resize(size_before_transf)(img)
            images[i] = tfm.ToTensor()(img)
        except Exception as e:
            raise ValueError(f"Error loading {qp}: {e}")

    # optional plotting of up to 6 random pairs
    if output_plots:
        os.makedirs("plots", exist_ok=True)
        pairs = list(enumerate(query_labels))
        pairs = [(i, lbl[0]) for i, lbl in pairs]
        sel = random.sample(pairs, min(len(pairs), 6))

        fig, axes = plt.subplots(len(sel), 2, figsize=(8, 3*len(sel)))
        if len(sel)==1:
            axes = [axes]
        for ax_row, (q_i, c_i) in zip(axes, sel):
            # query
            q_img = Image.open(query_paths[q_i])
            ax_row[0].imshow(q_img)
            ax_row[0].set_title("Query")
            ax_row[0].axis("off")
            # candidate
            c_img = Image.open(candidate_paths[c_i][0])
            ax_row[1].imshow(c_img)
            ax_row[1].set_title("Candidate")
            ax_row[1].axis("off")

        plt.tight_layout()
        out = get_incremented_filename("plots/query_candidate_pairs.png")
        plt.savefig(out)
        plt.close()
        print(f"Plot saved to {out}")

    return images, query_labels