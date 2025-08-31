# display_visual_clustering.py
import cv2
import numpy as np
import os
from collections import Counter
from ultralytics import YOLO

# --- Paste in the functions from your other files ---
# (Or import them if you have a proper project structure)

# From feature_extraction.py
from feature_extraction import get_resnet_embeddings
# From clustering.py
from clustering import get_spatial_clusters, get_visual_clusters
# From detection.py
from detection import process_image


# NEW: A helper function to create a visual collage of the clusters
def create_cluster_montage(image_crops, cluster_labels, optimal_k):
	"""
	Creates a single image displaying the items grouped by their visual cluster ID.
	"""
	# Group images by their assigned cluster label
	clusters = {i: [] for i in range(optimal_k)}
	for i, label in enumerate(cluster_labels):
		clusters[label].append(image_crops[i])

	# Define some parameters for the montage image
	max_imgs_per_row = 5
	thumb_h, thumb_w = 100, 100  # Thumbnail size
	padding = 10

	# Calculate the size of the output canvas
	max_rows = 0
	for k in clusters:
		rows = int(np.ceil(len(clusters[k]) / max_imgs_per_row))
		if rows > max_rows:
			max_rows = rows

	canvas_w = (thumb_w + padding) * optimal_k
	canvas_h = (thumb_h + padding) * max_rows + 50  # Add space for titles

	montage = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

	# Paste the thumbnails onto the canvas
	for cluster_id, crops in sorted(clusters.items()):
		col_x_start = (thumb_w + padding) * cluster_id

		# Add a title for the column
		cv2.putText(montage, f"Cluster {cluster_id}", (col_x_start + 5, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		for i, crop in enumerate(crops):
			row = i // max_imgs_per_row
			col = i % max_imgs_per_row
			y_start = (thumb_h + padding) * row + 50
			x_start = col_x_start + (thumb_w + padding) * col

			# Resize the crop to a standard thumbnail size
			thumbnail = cv2.resize(crop, (thumb_w, thumb_h))
			montage[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = thumbnail

	return montage


# --- Main Execution ---
if __name__ == '__main__':
	# --- Configuration ---
	MODEL_PATH = 'models/snapStockV3/best.pt'
	TEST_IMAGE_PATH = 'test_images_irl/tims-01.jpeg'  # <-- USE A GOOD IMAGE WITH MULTIPLE GROUPS

	# 1. Load the model
	yolo_model = YOLO(MODEL_PATH)

	# 2. Get initial detections and spatial clusters
	processed_data = process_image(yolo_model, TEST_IMAGE_PATH, conf=0.5)

	if processed_data and processed_data["boxes_object"].shape[0] > 0:
		spatial_labels, _ = get_spatial_clusters(processed_data["boxes_object"])

		# 3. Find the largest spatial group to test
		if len(spatial_labels) > 0:
			largest_group_id = Counter(spatial_labels).most_common(1)[0][0]
			print(f"Testing visual clustering on the largest spatial group: Group {largest_group_id}")

			# 4. Get the image crops for only that group
			boxes_resized = processed_data["boxes_object"].xyxy.cpu().numpy().astype(int)
			resized_image = processed_data["resized_image"]

			crops_for_group = []
			for i, label in enumerate(spatial_labels):
				if label == largest_group_id:
					x1, y1, x2, y2 = boxes_resized[i]
					crops_for_group.append(resized_image[y1:y2, x1:x2])

			# 5. Get embeddings for the crops
			embeddings = get_resnet_embeddings(crops_for_group)
			print(embeddings.shape)

			# 6. Run visual clustering
			visual_labels, k = get_visual_clusters(embeddings)
			print(f"K-Means found {k} optimal visual clusters within this spatial group.")

			# 7. Create and display the visual montage
			montage = create_cluster_montage(crops_for_group, visual_labels, k)
			cv2.imshow("Visual Clustering Results", montage)
			cv2.imwrite("visual_cluster_test.jpg", montage)
			print("Saved montage to 'visual_cluster_test.jpg'. Press any key to exit.")
			cv2.waitKey(0)
			cv2.destroyAllWindows()