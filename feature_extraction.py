import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


def get_resnet_embeddings(image_crops):
	"""
	Takes a list of cropped images and returns their ResNet50 feature embeddings.

	Args:
		image_crops (list): A list of cropped images as NumPy arrays (from OpenCV).

	Returns:
		A numpy array of feature embeddings, where each row is the embedding
		for one of the input image crops.
	"""
	# 1. Load a pre-trained ResNet50 model
	# We use the default weights pre-trained on the ImageNet dataset
	model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

	# 2. Modify the model to remove the final classification layer.
	# We want the output of the layer BEFORE the final prediction.
	feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

	# Set the model to evaluation mode
	feature_extractor.eval()

	# 3. Define the image preprocessing steps
	# These must match the format ResNet was originally trained on.
	preprocess = transforms.Compose([
		transforms.ToPILImage(),  # Convert NumPy array to PIL Image
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# 4. Process the images and extract embeddings
	embeddings = []
	# Ensure GPU is used if available
	device = torch.device("mps" if torch.mps.is_available() else "cpu")
	feature_extractor.to(device)

	with torch.no_grad():  # Disable gradient calculation for efficiency
		for crop in image_crops:
			# Apply the preprocessing transforms
			input_tensor = preprocess(crop)
			# Add a batch dimension (e.g., [3, 224, 224] -> [1, 3, 224, 224])
			input_batch = input_tensor.unsqueeze(0).to(device)

			# Get the feature vector from the model
			output = feature_extractor(input_batch)

			# Flatten the output tensor and move it to the CPU
			embeddings.append(output.squeeze().cpu().numpy())

	return np.array(embeddings)