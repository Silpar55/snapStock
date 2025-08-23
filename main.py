"""
	--> SnapStock <--

	Structure of the SnapStock backend:
	- Features:
		- Object Detection Model Using YOLO 8 and trained with COCO dataset
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