from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


# --- Private Helper Functions ---

def _relabel_noise_as_clusters(labels):
	"""
	Takes DBSCAN labels and re-labels noise points (-1) as their own unique clusters.
	"""
	if len(labels) == 0:
		return labels
	max_label = labels.max()
	next_label = max_label + 1
	final_labels = np.copy(labels)
	for i in range(len(final_labels)):
		if final_labels[i] == -1:
			final_labels[i] = next_label
			next_label += 1
	return final_labels


def _shrink_boxes(boxes_xyxy, scale_factor=0.5):
	"""
	Reduces the size of bounding boxes by a scale factor while keeping them centered.
	"""
	widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
	heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
	centers_x = boxes_xyxy[:, 0] + widths / 2
	centers_y = boxes_xyxy[:, 1] + heights / 2

	new_widths = widths * scale_factor
	new_heights = heights * scale_factor

	new_x1 = centers_x - new_widths / 2
	new_y1 = centers_y - new_heights / 2
	new_x2 = centers_x + new_widths / 2
	new_y2 = centers_y + new_heights / 2

	return np.stack((new_x1, new_y1, new_x2, new_y2), axis=1).astype(int)


def _boxes_to_corner_points(boxes_xyxy):
	"""Converts bounding boxes to a flat list of corners and a map to their original box index."""
	all_corners, box_indices = [], []
	for i, box in enumerate(boxes_xyxy):
		x1, y1, x2, y2 = box
		corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
		all_corners.extend(corners)
		box_indices.extend([i] * 4)
	return np.array(all_corners), np.array(box_indices)


def _aggregate_corner_labels(corner_labels, box_indices, num_boxes):
	"""Determines a single cluster label for each box by majority vote."""
	final_box_labels = []
	for i in range(num_boxes):
		labels_for_this_box = corner_labels[box_indices == i]
		if len(labels_for_this_box) > 0:
			most_common_label = Counter(labels_for_this_box).most_common(1)[0][0]
			final_box_labels.append(most_common_label)
		else:
			final_box_labels.append(-1)

	return np.array(final_box_labels)


# --- Main Public Functions ---

def get_spatial_clusters(boxes_object, min_samples=8, eps_multiplier=0.8, shrink_scale=0.5, size_filter_percentile=5):
	"""
	Main function for fully automated spatial clustering using the corner-point method.
	1. Filters out the smallest X% of boxes.
	2. Shrinks remaining boxes to improve separation.
	3. Uses an adaptive eps based on median box size to run DBSCAN on corner points.
	"""
	if boxes_object.shape[0] == 0:
		return np.array([]), boxes_object

	# --- Step 1: Filter Out Smallest Boxes by Percentile ---
	boxes_xyxy = boxes_object.xyxy.cpu().numpy()
	if len(boxes_xyxy) > 0 and size_filter_percentile > 0:
		areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
		area_cutoff = np.percentile(areas, size_filter_percentile)
		keep_indices = areas >= area_cutoff
		filtered_boxes_object = boxes_object[keep_indices]
		filtered_boxes_xyxy = filtered_boxes_object.xyxy.cpu().numpy()
	else:
		filtered_boxes_object = boxes_object
		filtered_boxes_xyxy = boxes_xyxy

	# --- Step 2: Cluster the FILTERED boxes  ---
	if filtered_boxes_xyxy.shape[0] * 4 < min_samples:  # Check against number of corner points
		# Not enough points to form a cluster, label each box individually
		cluster_labels = np.arange(filtered_boxes_xyxy.shape[0])
	else:
		shrunken_boxes = _shrink_boxes(filtered_boxes_xyxy, scale_factor=shrink_scale)
		corner_points, box_map = _boxes_to_corner_points(shrunken_boxes)

		# Calculate an adaptive eps based on the median size of the shrunken boxes
		box_widths = shrunken_boxes[:, 2] - shrunken_boxes[:, 0]
		box_heights = shrunken_boxes[:, 3] - shrunken_boxes[:, 1]
		box_sizes = (box_widths + box_heights) / 2
		median_size = np.median(box_sizes)
		adaptive_eps = median_size * eps_multiplier

		dbscan = DBSCAN(eps=adaptive_eps, min_samples=min_samples)
		corner_labels = dbscan.fit_predict(corner_points)

		aggregated_labels = _aggregate_corner_labels(corner_labels, box_map, len(filtered_boxes_xyxy))
		cluster_labels = _relabel_noise_as_clusters(aggregated_labels)

	# Return the labels and the actual boxes that were clustered
	return cluster_labels, filtered_boxes_object


def visualize_clusters(original_image, boxes_object, cluster_labels, scale_factors):
	"""
	Analyzes, summarizes, and draws color-coded bounding boxes on the original image.
	- Prints a summary of cluster sizes to the console.
	- Uses a deterministic color gradient for clusters.
	"""
	img_to_draw = original_image.copy()
	boxes_resized = boxes_object.xyxy.cpu().numpy()
	x_scale, y_scale = scale_factors

	# --- 1. Analyze and Summarize Clusters (New Feature) ---
	# Filter out labels for boxes that were removed (e.g., by size filter)
	valid_labels = [label for label in cluster_labels if label >= 0]

	if valid_labels:
		# Count how many items are in each group
		group_counts = Counter(valid_labels)
		# Invert the dictionary to group by size (e.g., {5 items: [group 0, group 1]})
		size_counts = {}
		for group_id, num_items in group_counts.items():
			if num_items not in size_counts:
				size_counts[num_items] = []
			size_counts[num_items].append(group_id)

		print("\n--- Cluster Analysis ---")
		print(f"Total groups found: {len(group_counts)}")
		# Sort sizes for a clean printout
		for size, groups in sorted(size_counts.items()):
			group_ids = ", ".join(map(str, sorted(groups)))
			group_str = "group" if len(groups) == 1 else "groups"
			item_str = "item" if size == 1 else "items"
			print(f"- {len(groups)} {group_str} with {size} {item_str} (Groups: {group_ids})")
		print("-" * 24)

	# --- 2. Generate Gradient Colors (New Feature) ---
	unique_labels = sorted([label for label in set(valid_labels)])
	num_clusters = len(unique_labels)
	colors = {}
	for i, label in enumerate(unique_labels):
		# Cycle through hues in the HSV color space for a rainbow gradient
		# OpenCV HSV Hue range is 0-179
		hue = int(i * (180 / num_clusters)) if num_clusters > 0 else 0
		hsv_color = np.uint8([[[hue, 255, 255]]])
		# Convert HSV to BGR for OpenCV
		bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
		colors[label] = tuple(map(int, bgr_color))

	# --- 3. Draw the Bounding Boxes ---
	for i, box_coords in enumerate(boxes_resized):
		cluster_id = cluster_labels[i]

		# Only draw boxes that were part of a valid cluster
		if cluster_id >= 0:
			x1, y1, x2, y2 = [int(c) for c in box_coords]

			# Scale coordinates to the original image size
			scaled_x1 = int(x1 * x_scale)
			scaled_y1 = int(y1 * y_scale)
			scaled_x2 = int(x2 * x_scale)
			scaled_y2 = int(y2 * y_scale)

			box_color = colors.get(cluster_id, (0, 0, 255))  # Default to red if missing

			cv2.rectangle(img_to_draw, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), box_color, 3)
			cv2.putText(img_to_draw, f"Group {cluster_id}", (scaled_x1, scaled_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
						box_color, 2)

	return img_to_draw
