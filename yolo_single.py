import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import onnxruntime as ort
import json # Added import for json

# Predict using YOLOv9 model
def predict_with_yolov9(image_path, model_path, labels_path, input_size):
    """
    Run inference using the converted YOLOv9 model on a single image.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the ONNX model file.
        labels_path (str): Path to the JSON file containing class labels.
        input_size (tuple): The expected input size (height, width) for the model.

    Returns:
        str: The predicted class label.
        PIL.Image.Image: The original loaded image.
    """
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # Load labels
    labels = load_json(labels_path)

    # Preprocess image
    original_image = Image.open(image_path).convert("RGB")
    image_resized = original_image.resize(input_size, Image.Resampling.BILINEAR)
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
    input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)

    # Load YOLOv9 model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name # Assuming classification output

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    predictions = outputs[0]

    # Postprocess predictions (assuming classification output)
    # Adapt this section if your model output is different (e.g., detection boxes)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[str(predicted_index)] # Assumes labels are indexed by string numbers

    return predicted_label, original_image

# Display prediction for a single image
def display_single_prediction(image_path, model_path, labels_path, input_size):
    """
    Predicts the class for a single image and displays the image with its prediction.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the ONNX model file.
        labels_path (str): Path to the JSON file containing class labels.
        input_size (tuple): The expected input size (height, width) for the model.
    """
    try:

        prediction, image = predict_with_yolov9(image_path, model_path, labels_path, input_size)
        print(prediction)

        # # Run prediction
        # prediction, img = predict_with_yolov9(image_path, model_path, labels_path, input_size)
        #
        # # Display image and prediction
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8)) # Create a single plot
        # ax.imshow(img)
        # ax.set_title(f"Prediction: {prediction}", fontsize=14)
        # ax.axis("off") # Hide axes ticks and labels
        #
        # plt.tight_layout()
        # plt.show()

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Main Execution ---

# Paths and parameters - **MODIFY THESE**
single_image_path = r"D:\code\nsfw_detection\downloaded_images\1735661205_306244774673770146_0.png"  # <--- Replace with the actual path to your image file
model_path = r"D:\code\nsfw_detection\nsfw_image_detection\falconsai_yolov9_nsfw_model_quantized.pt"    # <--- Replace with the actual path to your ONNX model
labels_path = r"D:\code\nsfw_detection\nsfw_image_detection\labels.json"        # <--- Replace with the actual path to your labels JSON file
input_size = (224, 224)                         # Standard input size, adjust if your model differs

# Check if the image file exists before proceeding (optional but recommended)
if os.path.exists(single_image_path):
    # Run prediction and display for the single image
    display_single_prediction(single_image_path, model_path, labels_path, input_size)
else:
    print(f"Error: The specified image file does not exist: {single_image_path}")