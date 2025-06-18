from collections import defaultdict
import torch
import os

def add_queries_prefix(path):
    prefix = "data/queries/"
    if not path.startswith(prefix):
        return prefix + path
    return path

def build_and_save_candidate_to_queries(
    intersections_pt_path: str,
    filtered_queries_pt_path: str,
    output_pt_path: str,
    iou_threshhold: float = 0.5
):
    """
    Loads intersections and filtered queries from .pt files,
    builds the inverted index (candidate_path -> [query_paths]),
    and saves it to output_pt_path.
    The output filename will include the IoU threshold.
    Also prints how many queries (not necessarily unique) are preserved in the final dict,
    and how many unique queries are preserved.
    """
    # Format the threshold for the filename (e.g., 0.5 -> 0.50)
    iou_str = f"{iou_threshhold:.2f}"
    base, ext = os.path.splitext(output_pt_path)
    output_pt_path_with_thresh = f"{base}_iou_{iou_str}{ext}"

    # Load the data
    q_db_intersections = torch.load(intersections_pt_path)
    filtered_queries = torch.load(filtered_queries_pt_path)
    print('There are', len(filtered_queries), 'filtered queries in total')

    # Build the inverted index
    candidate_to_queries = defaultdict(list)
    for q_path in filtered_queries:
        positives = q_db_intersections.get(str(q_path), [])
        for candidate_path in positives:
            if float(candidate_path[3]) > iou_threshhold:
                candidate_to_queries[str(candidate_path[0])].append(str(q_path))

    # Count total (not unique) queries preserved in the final dict
    total_queries_preserved = sum(len(query_list) for query_list in candidate_to_queries.values())
    print(f"Total number of queries (not necessarily unique) preserved in the final dict: {total_queries_preserved}")

    # Count unique queries preserved
    all_queries = []
    for query_list in candidate_to_queries.values():
        all_queries.extend(query_list)
    unique_queries = set(all_queries)
    print(f"Number of unique queries preserved in the final dict: {len(unique_queries)}")

    # Save the inverted index
    torch.save(dict(candidate_to_queries), output_pt_path_with_thresh)
    print(f"Inverted index saved to {output_pt_path_with_thresh}")

# Example usage
build_and_save_candidate_to_queries(
    "./data/queries_intersections_with_db_2021.pt",
    "./data/filtered_queries_not_in_test_and_val.pt",
    "./data/candidate_to_queries_with_db_2021.pt",
    iou_threshhold=0.0
)