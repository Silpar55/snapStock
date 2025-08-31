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
	boxes_object = results.boxes[high_conf_indices]
	# Calculate scaling factors
	x_scale = original_width / target_size[0]
	y_scale = original_height / target_size[1]

	return {
		"original_image": original_image,  # Visualization purposes
		"resized_image": resized_image,
		"boxes_object": boxes_object,  # DBSCAN purpose
		"scale_factors": (x_scale, y_scale)  # Rescaling purpose
	}


def visualize_detections(data):
	"""
	Draws the detected bounding boxes and labels on the original image.
	"""
	# Unpack the data
	image_to_draw = data["original_image"].copy()
	boxes_object = data["boxes_object"]
	x_scale, y_scale = data["scale_factors"]

	# Loop through each detected box
	for box in boxes_object:
		# Get coordinates from the resized image space
		x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

		# Scale coordinates back to the original image size
		scaled_x1 = int(x1 * x_scale)
		scaled_y1 = int(y1 * y_scale)
		scaled_x2 = int(x2 * x_scale)
		scaled_y2 = int(y2 * y_scale)

		# Get confidence and class info
		confidence = float(box.conf[0])
		label = f"Item {confidence:.2f}"

		# Draw the bounding box and label
		cv2.rectangle(image_to_draw, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)
		cv2.putText(image_to_draw, label, (scaled_x1, scaled_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	return image_to_draw
