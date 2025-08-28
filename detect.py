import cv2
import os
from ultralytics import YOLO

# --- Setup ---
# Load your custom trained model
model = YOLO("models/snapStockV3/best.pt")

# Define the input and output folders
input_folder = "test_images_irl/"
output_folder = "test_results_irl/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# --- Get a list of all image files in the input folder ---
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- Process Images in a Batch ---
# Create full paths for the model
full_image_paths = [os.path.join(input_folder, f) for f in image_files]

if full_image_paths:
	# Run inference on the entire batch of images
	results_list = model(full_image_paths)
	print(f"Found {len(image_files)} images. Processing...")

	# --- Loop through results and draw on each image ---
	# The results_list will be in the same order as the input_image_paths
	for i, results in enumerate(results_list):
		# Get the original image path
		original_img_path = full_image_paths[i]

		# Load the original image with OpenCV
		img = cv2.imread(original_img_path)

		print(f"  -> Detecting objects in {os.path.basename(original_img_path)}")

		# Iterate through each detected box in the current image's results
		for box in results.boxes:
			# Get coordinates
			x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

			# Get the confidence score
			confidence = float(box.conf[0])

			if confidence > 0.5:
				# Get the class ID and name
				class_id = int(box.cls[0])
				class_name = results.names[class_id]

				# Create the label text
				label = f"{class_name} {confidence:.2f}"

				# Draw the bounding box on the image (green color, thickness 2)
				cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

				# Put the label text above the bounding box
				cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		# --- Save the Resulting Image ---
		# Construct the output path
		output_path = os.path.join(output_folder, os.path.basename(original_img_path))

		# Save the image with the drawings
		cv2.imwrite(output_path, img)
		print(f"     ... Saved result to {output_path}")


print("\nProcessing complete.")