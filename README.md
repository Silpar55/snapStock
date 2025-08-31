# SnapStock 📸

An AI-powered inventory management system that uses computer vision to automatically detect, cluster, and catalog products for small businesses, starting with a focus on bakeries.

## About The Project

Many small businesses lack a proper database to track their inventory, leading to inefficiencies and lost revenue. Manually cataloging every item is a tedious and time-consuming process. SnapStock solves this problem by providing an intelligent system that can build an inventory database from a few simple photos.

This project is initially focused on the bakery sector to establish a solid framework that can be adapted to other industries in the future.

***
## Project Status
**Status:** ✅ The core computer vision pipeline is complete. All modules for detection, feature extraction, and clustering have been developed and tested. The project is now moving into the final backend integration phase.

***
## The AI Pipeline
The SnapStock pipeline processes an image in several automated stages to produce logical groups of items ready for user labeling:

1.  **Image Processing:** User-uploaded images are standardized by resizing them to a consistent resolution (640x640) to ensure reliable performance.

2.  **Class-Agnostic Detection (YOLOv11):** A fine-tuned YOLOv11 model detects all relevant objects in the image, labeling them with a single, universal class: `item`. This makes the detector highly robust at *finding* objects without needing to distinguish between them yet.

3.  **Spatial Clustering (Adaptive DBSCAN):** The detected `item`s are then clustered based on their physical location on the shelf. This step uses a custom DBSCAN implementation with pre-processing (box shrinking) and an adaptive `eps` parameter to intelligently identify distinct physical groups (e.g., items on a specific tray).

4.  **Visual Clustering (ResNet + K-Means):** For each spatial group, a pre-trained ResNet50 model extracts a visual 'fingerprint' (embedding) for each item. The K-Means algorithm then clusters these fingerprints to separate the items into visually similar groups (e.g., all croissants vs. all muffins).

The final output is a structured set of visually and spatially distinct groups, which are then passed to the user for final labeling.

***
## Next Steps
The final stages of the project involve:

1.  **Database Generation (`db_generator.py`):** Implementing the final step of the AI pipeline, where user-provided labels for the visual clusters are sent to the Gemini API to generate a database schema.
2.  **Backend API Development:** Wrapping all the completed Python scripts (`detection.py`, `clustering.py`, `feature_extraction.py`) into a robust FastAPI application with the endpoints outlined below.

### API Endpoints
* `/upload`: Handles bulk photo uploads from the user.
* `/process`: Triggers the full AI pipeline and returns the clustered image groups to the frontend.
* `/generate-db`: Takes the user-provided labels and calls the LLM to generate the final database script.

***
## The Dataset
Due to the lack of large, pre-existing datasets, a custom dataset was built from scratch. The final version was annotated with a single `item` class to train the robust, class-agnostic detector.

The public Roboflow project can be found here: **[SnapStock Bakery Detection](https://app.roboflow.com/snapstock/snapstock-3.0-eysja/2)**

***
## Continuing to Train the Model
If you wish to improve the model or adapt it for a new set of products, you have two options to get started with training.

### Option 1: Using Roboflow
This is the easiest way to manage the dataset and train the model.

1.  Navigate to the public Roboflow project URL provided above and fork the dataset.
2.  Add new images, annotate them, and generate a new version.
3.  Use Roboflow's integrated notebooks to fine-tune the model on your new dataset version.

### Option 2: Using Google Colab and Google Drive
This method is ideal if you have a copy of the dataset and want to run the training process manually.

1.  **Prepare the Dataset:** Unzip the provided dataset folder on your local machine.
2.  **Upload to Google Drive:** Upload the entire unzipped folder to your Google Drive.
3.  **Open Google Colab:** Launch the provided training notebook (`train_model.ipynb`) in Google Colab.
4.  **Connect to GPU:** Ensure your runtime is connected to a GPU.
5.  **Mount Google Drive:** Run the first code cell to mount your Google Drive.
6.  **Update the Data Path:** Update the `data` parameter in the `model.train()` call to point to the `data.yaml` file in your Google Drive.
7.  **Run Training:** Execute the training cells. Your new model weights will be saved in your Google Drive.