"""
	--> SnapStock V 1.1 Update <--

	NOTE: The project idea is generalized and will be hard to achieve it doing that
	for every sector, therefore, we are going to do it for a bakery store, in that way we
	can follow the structure and adjust the dataset for any sector.

    DATASET: Due to the lack of large, pre-existing datasets for bakery goods,
    a custom dataset was built from scratch. A Python script was used to programmatically
    download hundreds of images for a comprehensive list of categories (croissants, muffins, etc.)
    from web search results. This raw dataset then underwent a crucial manual cleaning phase to
    remove irrelevant or low-quality images. The curated images were then annotated with bounding
    boxes using Roboflow. Finally, the dataset was generated with preprocessing (resize to 640x640)
    and data augmentation (flips, rotations, brightness adjustments) to create a robust,
    training-ready collection.

    DATASET - Roboflow: https://app.roboflow.com/snapstock/bakery-detection-evzzv/models

	Structure of the SnapStock backend:
	- Features:
		- Object Detection Model Using YOLO 11 and trained with COCO dataset
		- Features Extraction using ResNet50
		- Clustering using K-means
		- Database Generation using Gemini API

	The web API will be created using FastAPI:

	- Endpoints:
		- /upload: An endpoint to handle the user's bulk photo upload.
		- /process: This endpoint will trigger the AI pipeline (detect, extract, cluster)
		and return the clustered image groups to the frontend.
		- /generate-db: This endpoint takes the user's labels for the clusters and calls
		the LLM script to generate and return the final database script.
"""
import os

import cv2
from detect import detect_and_filter_objects
from clustering import coords_to_midpoints, visualize_clusters
from sklearn.cluster import DBSCAN

"""
	This file is for testing any new features added 
	following the workflow of the project
"""

img_path = "./test_images_irl/tims-01.jpeg"
original_image = cv2.imread(img_path)
"""
	Detecting objects in the image and getting their bounding boxes based
	on confidence score (default 0.5)
"""

results = detect_and_filter_objects(img_path)

"""
	Calculate the center coordinates of the bounding boxes to apply
	DBSCAN for clustering
"""

midpoints_results = []

for boxes in results:
	midpoints = coords_to_midpoints(boxes)
	midpoints_results.append(midpoints)

"""
	Applying DBSCAN algorithm
"""

cluster = DBSCAN(eps=100, min_samples=1)

for i, midpoints in enumerate(midpoints_results):
	cluster_labels = cluster.fit_predict(midpoints)

	print("Visualizing results...")
	output_image = visualize_clusters(original_image.copy(), results[i], cluster_labels)

	# Construct the output filename
	base_name = os.path.splitext(os.path.basename('./test_images_irl/tims_DBSCAN.jpeg'))[0]
	output_filename = f"{base_name}_clustered.jpg"

	# Save the final image
	cv2.imwrite(output_filename, output_image)
	print(f"Successfully saved clustered image to: {output_filename}")

