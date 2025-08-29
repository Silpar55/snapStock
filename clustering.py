"""
	Giving a vector of bounding boxes of an image
	return a list of their midpoints coordinates
"""
import random

import cv2


def coords_to_midpoints(boxes):
	midpoints = []
	for box in boxes:
		x1, y1, x2, y2 = box

		midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

		midpoints.append(midpoint)

	return midpoints


def visualize_clusters(image, boxes, cluster_labels):
	"""
	Draws color-coded bounding boxes on an image based on their cluster labels.
	"""
	# Generate a unique color for each cluster ID
	unique_labels = set(cluster_labels)
	colors = {}
	for label in unique_labels:
		if label == -1:
			# Use a specific color for noise/outliers (e.g., white)
			colors[label] = (255, 255, 255)
		else:
			# Generate a random color for each valid cluster
			colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

	# Draw the bounding boxes
	for i, box_coords in enumerate(boxes):
		x1, y1, x2, y2 = [int(c) for c in box_coords]
		cluster_id = cluster_labels[i]

		# Get the color for the current cluster
		box_color = colors[cluster_id]

		# Draw the rectangle
		cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
		# Put the cluster ID label on the box
		cv2.putText(image, f"Group {cluster_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

	return image

