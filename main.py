from ultralytics import YOLO
import cv2
import os
import logging
from detect import process_image, visualize_detections
from clustering import get_spatial_clusters, get_visual_clusters , visualize_clusters
from feature_extraction import get_resnet_embeddings


# --- 1. Configuration ---
logging.basicConfig(level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{')
CONFIG = {
	"model_path": 'models/snapStockV3/best.pt',
	"image_path": 'test_images_irl/tims-01.jpeg',
	"confidence_threshold": 0.4,
	"target_size": (640, 640),
	"size_filter_percentile": 5,
	"dbscan_params": {"min_samples": 8, "eps_multiplier": 2.2, "shrink_scale": 0.4},
	"output_dir": "test_dbscan_images/"
}

# --- 2. Main Execution ---
if __name__ == '__main__':

	# --- 3. Load Model and create output directory

	model = YOLO(CONFIG["model_path"])
	os.makedirs(CONFIG["output_dir"], exist_ok=True)

	# --- 4. Get boxes for the image

	processed_data = process_image(
		model, CONFIG["image_path"],
		conf=CONFIG["confidence_threshold"], target_size=CONFIG["target_size"]
	)

	if processed_data and processed_data["boxes_object"].shape[0] > 0:

		# final_image = visualize_detections(processed_data)
		# cv2.imshow("YOLO Detections", final_image)
		# cv2.waitKey(0)  # Wait for a key press to close the window
		# cv2.destroyAllWindows()

		# --- 5. Clustering ---
		logging.info("--- Processing clustering ---")

		cluster_labels, filtered_boxes_object = get_spatial_clusters(
			processed_data["boxes_object"],
			size_filter_percentile=CONFIG["size_filter_percentile"],
			**CONFIG["dbscan_params"]
		)

		# image = visualize_clusters(
		# 	processed_data["original_image"],
		# 	filtered_boxes_object,
		# 	cluster_labels,
		# 	processed_data["scale_factors"]
		# )
		# output_path = os.path.join(CONFIG["output_dir"], CONFIG["image_path"].split('/')[-1])
		# cv2.imwrite(output_path, image)
		# logging.info(f"Saved corner point result to {output_path}")

		# --- 6. Feature Extraction---
		image_crops = []
		boxes_xyxy = filtered_boxes_object.xyxy.cpu().numpy().astype(int)
		for box_coords in boxes_xyxy:
			x1, y1, x2, y2 = box_coords
			crop = processed_data["resized_image"][y1:y2, x1:x2]
			image_crops.append(crop)

		if image_crops:
			embeddings = get_resnet_embeddings(image_crops)
			print(f"Successfully generated {len(embeddings)} embeddings.")

			print(get_visual_clusters(embeddings))

	else:
		logging.info("No objects were detected in the image.")
