import torch
from datasets import utils

# List of POIs with (name, lat, lon)
test_and_eval_sets = [
    ("Alps", 45, 10),
    ("Texas", 30, -95),
    ("Toshka Lakes", 23, 30),
    ("Amazon", -3, -60),
    ("Napa", 38, -122),
    ("Gobi", 40, 105),
]

DIST_THRESHOLD = 2500  # km



def get_center_from_footprint(footprint):
    """
    Given a flat tuple: (lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4),
    return the center as (lat, lon).
    """
    lats = [footprint[i] for i in range(0, 8, 2)]
    lons = [footprint[i] for i in range(1, 8, 2)]
    return (sum(lats) / 4, sum(lons) / 4)


def filter_images_by_POI_distance(paths, points_of_interest, dist_threshold):
    """
    Returns two lists:
    - paths farther than dist_threshold from ALL POIs
    - paths within dist_threshold of ANY POI
    """
    # Get [N, 2] tensor of [lat, lon] for each path
    textcoords = torch.tensor(
    [get_center_from_footprint(utils.get_footprint_from_path(p)) for p in paths],
    dtype=torch.float32
)

    # Get [M, 2] tensor of [lat, lon] for each POI
    poi_coords = torch.tensor([[lat, lon] for _, lat, lon in points_of_interest], dtype=torch.float32)

    # For each path, check if it is farther than the threshold from ALL POIs
    keep_mask = torch.ones(len(paths), dtype=torch.bool)
    for poi_coord in poi_coords:
        poi_expanded = poi_coord.unsqueeze(0).expand(textcoords.shape[0], -1)
        distances = utils.batch_geodesic_distances(poi_expanded, textcoords)
        keep_mask &= (distances > dist_threshold)

    # Invert mask for those within threshold of ANY POI
    in_test_val_mask = ~keep_mask

    filtered_paths = [p for p, keep in zip(paths, keep_mask) if keep]
    in_test_val_paths = [p for p, in_tv in zip(paths, in_test_val_mask) if in_tv]
    return filtered_paths, in_test_val_paths

def main():
    # Load the intersection dictionary
    q_db_intersections = torch.load(
        "./data/queries_intersections_with_db_2021.pt", weights_only=False
    )
    queries_paths = list(q_db_intersections.keys())

    # Filter
    filtered_paths, in_test_val_paths = filter_images_by_POI_distance(
        queries_paths, test_and_eval_sets, DIST_THRESHOLD
    )

    # Save filtered paths
    torch.save(filtered_paths, "./data/filtered_queries_not_in_test_and_val.pt")
    print(f"Saved {len(filtered_paths)} filtered paths to filtered_queries_not_in_test_and_val.pt")

    # Save in-test/val paths
    torch.save(in_test_val_paths, "./data/filtered_queries_in_test_and_val.pt")
    print(f"Saved {len(in_test_val_paths)} test/val paths to filtered_queries_in_test_and_val.pt")

if __name__ == "__main__":
    main()