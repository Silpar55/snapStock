import os
from detect import detect_and_filter_objects, test_model
from clustering import coords_to_midpoints, coords_to_corners, visualize_clusters
import cv2
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# --- 1. Configuration ---

MODEL_PATH = "models/snapStockV3/best.pt"
TEST_IMAGE_PATH = "./test_images_irl/tims-01.jpeg"
CONFIDENCE_THRESHOLD = 0.5

# --- 2. Load Model and Image ---
model = YOLO(MODEL_PATH)
original_image = cv2.imread(TEST_IMAGE_PATH)


# --- 3. Run Detect.py ---
results = detect_and_filter_objects(model, TEST_IMAGE_PATH, CONFIDENCE_THRESHOLD)

test_model(model)


# --- 4. Run Clustering.py
#
# midpoints_results = []
# corners_results = []  # Testing with corners instead of midpoints
#
# for boxes in results:
# 	midpoints = coords_to_midpoints(boxes)
# 	midpoints_results.append(midpoints)
#
# for boxes in results:
# 	corners = coords_to_corners(boxes)
# 	corners_results.append(corners)
#
# """
# 	Applying DBSCAN algorithm
# """
#
# cluster_midpoints = DBSCAN(eps=100, min_samples=1)
# cluster_corners = DBSCAN(eps=200, min_samples=6)
#
# for i, midpoints in enumerate(midpoints_results):
# 	cluster_labels = cluster_midpoints.fit_predict(midpoints)
#
# 	print("Visualizing results...")
# 	output_image = visualize_clusters(original_image.copy(), results[i], cluster_labels)
#
# 	# Construct the output filename
# 	base_name = os.path.splitext(os.path.basename('./test_dbscan_images/tims_DBSCAN_midpoints.jpeg'))[0]
# 	output_filename = f"{base_name}_clustered.jpg"
#
# 	# Save the final image
# 	cv2.imwrite(output_filename, output_image)
# 	print(f"Successfully saved clustered image to: {output_filename}")
#
# for i, corners in enumerate(corners_results):
# 	cluster_labels = cluster_corners.fit_predict(corners)
#
# 	print("Visualizing results...")
# 	output_image = visualize_clusters(original_image.copy(), results[i], cluster_labels, is_corner_method=True)
#
# 	# Construct the output filename
# 	base_name = os.path.splitext(os.path.basename('./test_dbscan_images/tims_DBSCAN_corners.jpeg'))[0]
# 	output_filename = f"{base_name}_clustered.jpg"
#
# 	# Save the final image
# 	cv2.imwrite(output_filename, output_image)
# 	print(f"Successfully saved clustered image to: {output_filename}")

