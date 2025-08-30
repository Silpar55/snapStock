import cv2
import random
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def _relabel_noise_as_clusters(labels):
	"""
	Takes DBSCAN labels and re-labels noise points (-1) as their own unique clusters.
	This is because some valid groups have only one object
	"""
	if len(labels) == 0:
		return labels

	# Find the highest existing cluster ID
	max_label = labels.max()

	# Start the new label counter from the next number
	new_label_counter = max_label + 1

	# Create a new array for the final labels
	final_labels = np.copy(labels)

	# Loop through and replace each -1 with a new, unique cluster ID
	for i in range(len(final_labels)):
		if final_labels[i] == -1:
			final_labels[i] = new_label_counter
			new_label_counter += 1

	return final_labels


def boxes_to_midpoints(boxes_xyxy):
	"""Takes a list of bounding boxes and returns their midpoints."""
	return [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes_xyxy]


def boxes_to_corner_points(boxes_xyxy):
	"""Converts bounding boxes to a flat list of corners and a map to their original box index."""
	all_corners, box_indices = [], []
	for i, box in enumerate(boxes_xyxy):
		x1, y1, x2, y2 = box
		corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
		all_corners.extend(corners)
		box_indices.extend([i] * 4)
	return np.array(all_corners), np.array(box_indices)


def aggregate_corner_labels(corner_labels, box_indices, num_boxes):
	"""
	Determines a single cluster label for each box based on the
	most common label among its four corner points (majority vote).
	"""
	final_box_labels = []
	for i in range(num_boxes):
		# Find the labels for the 4 corners of the current box
		labels_for_this_box = corner_labels[box_indices == i]
		if len(labels_for_this_box) > 0:
			# Find the most common label
			most_common_label = Counter(labels_for_this_box).most_common(1)[0][0]
			final_box_labels.append(most_common_label)
		else:
			final_box_labels.append(-1)  # Default to noise
	return np.array(final_box_labels)


def run_dbscan_on_midpoints(boxes_xyxy, eps, min_samples):
	"""Performs the full midpoint DBSCAN workflow."""
	midpoints = boxes_to_midpoints(boxes_xyxy)
	if not midpoints:
		return np.array([])
	dbscan = DBSCAN(eps=eps, min_samples=min_samples)
	labels = dbscan.fit_predict(midpoints)

	# Post-process the labels to handle single-item groups
	labels = _relabel_noise_as_clusters(labels)

	return labels


def run_dbscan_on_corners(boxes_xyxy, eps, min_samples):
	"""Performs the full corner-point DBSCAN workflow."""
	corner_points, box_map = boxes_to_corner_points(boxes_xyxy)
	if corner_points.size == 0:
		return np.array([])
	dbscan = DBSCAN(eps=eps, min_samples=min_samples)
	corner_labels = dbscan.fit_predict(corner_points)
	aggregate_labels = aggregate_corner_labels(corner_labels, box_map, len(boxes_xyxy))

	# Post-process the labels to handle single-item groups
	final_labels = _relabel_noise_as_clusters(aggregate_labels)

	return final_labels


def visualize_clusters_on_original(image, boxes_resized, scale_factors, cluster_labels):
	"""Draws color-coded boxes (scaled to original size) on the original image."""
	# This function works for both methods as it just needs the final labels per box
	x_scale, y_scale = scale_factors
	unique_labels = set(cluster_labels)
	colors = {label: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for label in
			  unique_labels if label != -1}
	colors[-1] = (200, 200, 200)

	for i, box_coords in enumerate(boxes_resized):
		x1, y1, x2, y2 = [int(c) for c in box_coords]

		scaled_x1 = int(x1 * x_scale)
		scaled_y1 = int(y1 * y_scale)
		scaled_x2 = int(x2 * x_scale)
		scaled_y2 = int(y2 * y_scale)

		cluster_id = cluster_labels[i]
		box_color = colors[cluster_id]

		cv2.rectangle(image, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), box_color, 3)
		cv2.putText(image, f"Group {cluster_id}", (scaled_x1, scaled_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color,
					2)

	return image


def find_optimal_eps(points, min_samples):
	"""
	Calculates and plots the k-distance graph to help find the optimal eps value.
	'k' is equal to min_samples.
	"""
	if len(points) < min_samples:
		print("Not enough points to find optimal eps. Need at least {min_samples} points.")
		return

	# Find the distance to the k-th nearest neighbor for each point
	k = min_samples
	neighbors = NearestNeighbors(n_neighbors=k)
	neighbors_fit = neighbors.fit(points)
	distances, indices = neighbors_fit.kneighbors(points)

	# Get the distances to the k-th neighbor (the last column) and sort them
	k_distances = sorted(distances[:, k - 1], reverse=True)

	# Plot the k-distance graph
	plt.figure(figsize=(10, 6))
	plt.plot(k_distances)
	plt.title(f'K-distance Graph (k={k})')
	plt.xlabel("Points (sorted by distance)")
	plt.ylabel(f"Distance to {k}-th Nearest Neighbor (eps)")
	plt.grid(True)
	print("Displaying the K-distance graph. Look for the 'elbow' and use that y-value for your 'eps'.")
	print("Close the plot window to continue the script.")
	plt.show()
