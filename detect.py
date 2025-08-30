import os
import cv2


def process_image(model, image_path, conf=0.5, target_size=(640, 640)):
	"""
	Loads an image, resizes it, runs detection, and returns the resized
	boxes along with the factors needed to scale them back to the original size.
	"""
	original_image = cv2.imread(image_path)
	if original_image is None:
		print(f"Warning: Could not read image at {image_path}")
		return None

	original_height, original_width, _ = original_image.shape

	resized_image = cv2.resize(original_image, target_size)

	results = model(resized_image)[0]

	high_conf_indices = results.boxes.conf > conf
	filtered_boxes_resized = results.boxes[high_conf_indices].xyxy.cpu().numpy().astype(int)

	# Calculate scaling factors
	x_scale = original_width / target_size[0]
	y_scale = original_height / target_size[1]

	return {
		"original_image": original_image,  # Visualization purposes
		"resized_image": resized_image,
		"boxes_object": filtered_boxes_resized,  # DBSCAN purpose
		"scale_factors": (x_scale, y_scale)  # Rescaling purpose
	}


def test_model(model, input_folder="test_images_irl/", output_folder="test_results_irl/"):
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
