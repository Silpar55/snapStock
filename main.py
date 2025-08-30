from ultralytics import YOLO
import cv2
import os
import logging
from detect import process_image, test_model
from clustering import run_dbscan_on_corners, run_dbscan_on_midpoints, visualize_clusters_on_original, find_optimal_eps, \
	boxes_to_corner_points, boxes_to_midpoints

# --- 1. Configuration ---
logging.basicConfig(level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{')
CONFIG = {
    "model_path": 'models/snapStockV3/best.pt',
    "image_path": 'test_images_irl/tims-01.jpeg',
    "confidence_threshold": 0.5,
    "target_size": (640, 640),
    "midpoint_params": {"eps": 65, "min_samples": 2},
    "corner_params": {"eps": 50, "min_samples": 6},
    "output_dir": "test_dbscan_images/"
}


# --- 2. Main Execution ---
if __name__ == '__main__':

	# --- 3. Load Model and create output directory

	model = YOLO(CONFIG["model_path"])
	os.makedirs(CONFIG["output_dir"], exist_ok=True)

	test_model(model)

	processed_data = process_image(
		model, CONFIG["image_path"],
		conf=CONFIG["confidence_threshold"], target_size=CONFIG["target_size"]
	)

	if processed_data and processed_data["boxes_object"].shape[0] > 0:
		boxes_resized = processed_data["boxes_object"]

		# --- Run Corner Point Method ---
		logging.info("--- Processing with Corner Point Method ---")
		# corner_points, _ = boxes_to_corner_points(boxes_resized)
		# find_optimal_eps(points=corner_points, min_samples=CONFIG["corner_params"]["min_samples"])

		corner_labels = run_dbscan_on_corners(boxes_resized, **CONFIG["corner_params"])
		corner_image = visualize_clusters_on_original(
			processed_data["original_image"].copy(),
			boxes_resized,
			processed_data["scale_factors"],
			corner_labels
		)
		corner_output_path = os.path.join(CONFIG["output_dir"], "result_corners.jpg")
		cv2.imwrite(corner_output_path, corner_image)
		logging.info(f"Saved corner point result to {corner_output_path}")

		# --- Run Midpoint Method ---
		logging.info("--- Processing with Midpoint Method ---")
		# center_points = boxes_to_midpoints(boxes_resized)
		# find_optimal_eps(points=center_points, min_samples=CONFIG["midpoint_params"]["min_samples"])
		midpoint_labels = run_dbscan_on_midpoints(boxes_resized, **CONFIG["midpoint_params"])
		midpoint_image = visualize_clusters_on_original(
			processed_data["original_image"].copy(),
			boxes_resized,
			processed_data["scale_factors"],
			midpoint_labels
		)
		midpoint_output_path = os.path.join(CONFIG["output_dir"], "result_midpoints.jpg")
		cv2.imwrite(midpoint_output_path, midpoint_image)
		logging.info(f"Saved midpoint result to {midpoint_output_path}")
	else:
		logging.info("No objects were detected in the image.")
