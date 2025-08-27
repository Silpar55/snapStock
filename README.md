# SnapStock 📸

An AI-powered inventory management system that uses computer vision to automatically detect, cluster, and catalog products for small businesses, starting with a focus on bakeries.

***
### Developer's Note: Project Evolution & Next Steps

* **Dataset Refinement (Solving "Domain Shift"):** It was discovered that training the model on isolated product images resulted in poor performance on real-world shelf photos. To address this, the project is moving to a new dataset composed of realistic, cluttered shelf images. This will ensure the model is trained in an environment that mirrors its final use case, significantly improving its accuracy and robustness.

* **Enhanced Clustering Logic (Spatial + Visual):** The AI pipeline is being upgraded to a two-stage clustering process for a more intuitive user experience.
    1.  **Spatial Clustering:** First, a **DBSCAN** algorithm will group detected items based on their physical proximity on the shelf.
    2.  **Visual Clustering:** Then, within each of those spatial groups, **ResNet and K-Means** will cluster items by their visual similarity. This allows the application to present logical groups to the user (e.g., "all the items on the top-left tray") rather than just visually similar items from all over the store.
***

## About The Project

Many small businesses lack a proper database to track their inventory, leading to inefficiencies and lost revenue. Manually cataloging every item is a tedious and time-consuming process. SnapStock solves this problem by providing an intelligent system that can build an inventory database from a few simple photos.

This project is initially focused on the bakery sector to establish a solid framework that can be adapted to other industries in the future.

***
## Tech Stack & Features
The SnapStock backend is built with a powerful set of modern AI and web technologies:

* **Object Detection:** **YOLOv8** fine-tuned on a custom dataset to locate products in images.
* **Feature Extraction:** **ResNet50** to create a unique numerical "fingerprint" for each detected object.
* **Clustering:** **Scikit-learn (K-Means)** to group similar objects based on their features.
* **Database Generation:** **Gemini API** to generate a SQL or MongoDB script from the final labeled product groups.
* **Web Framework:** **FastAPI** to serve the AI pipeline through a robust and fast API.

***
## API Endpoints
The backend provides the following endpoints:
* `/upload`: Handles bulk photo uploads from the user.
* `/process`: Triggers the AI pipeline (detect, extract, cluster) and returns the clustered image groups to the frontend for labeling.
* `/generate-db`: Takes the user-provided labels and calls the LLM to generate the final database script.

***
## The Dataset
Due to the lack of large, pre-existing datasets for bakery goods, a custom dataset was built from scratch.

1.  **Data Collection:** A Python script was used to programmatically download hundreds of images for a comprehensive list of categories (croissants, muffins, etc.) from web search results.
2.  **Data Cleaning:** The raw dataset then underwent a crucial manual cleaning phase to remove irrelevant or low-quality images.
3.  **Annotation & Processing:** The curated images were annotated with bounding boxes using Roboflow. The final dataset was generated with preprocessing (resize to 640x640) and data augmentation (flips, rotations, etc.) to create a robust, training-ready collection.

The public Roboflow project can be found here: **[SnapStock Bakery Detection](https://app.roboflow.com/snapstock/bakery-detection-evzzv/models)**

***
## Continuing to Train the Model
If you wish to improve the model or adapt it for a new set of products, you have two options to get started with training.

### Option 1: Using Roboflow
This is the easiest way to manage the dataset and train the model.

1.  Navigate to the public Roboflow project URL provided above.
2.  You can "fork" the dataset to your own Roboflow account.
3.  From there, you can add new images, annotate them, and generate a new version of the dataset.
4.  Roboflow provides integrated training notebooks (including Google Colab) that can be used to fine-tune the model on your new dataset version with just a few clicks.

### Option 2: Using Google Colab and Google Drive
This method is ideal if you have a copy of the dataset and want to run the training process manually.

1.  **Prepare the Dataset:** Unzip the provided dataset folder (e.g., `Bakery-Detection.zip`) on your local machine.
2.  **Upload to Google Drive:** Upload the entire unzipped `Bakery-Detection` folder to your Google Drive.
3.  **Open Google Colab:** Launch the provided training notebook (e.g., `train_model.ipynb`) in Google Colab.
4.  **Connect to GPU:** Ensure your Colab runtime is connected to a GPU (**Runtime** -> **Change runtime type** -> **T4 GPU**).
5.  **Mount Google Drive:** The first code cell in the notebook will allow you to mount your Google Drive, giving the notebook access to your files.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
6.  **Update the Data Path:** The most important step is to tell the training script where to find your dataset. Find the line in the script that calls `model.train()` and update the `data` parameter to point to the `data.yaml` file inside your Google Drive.
    ```python
    # Example path, yours might be slightly different
    data_path = '/content/drive/MyDrive/Bakery-Detection/data.yaml'

    results = model.train(data=data_path, device=0, ...)
    ```
7.  **Run Training:** Execute the training cells to fine-tune the model on your dataset. Your new model weights (`best.pt`) will be saved in your Google Drive.