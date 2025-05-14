import os
import cv2
import pandas as pd
import numpy as np

def images_to_csv_with_metadata(image_folder, output_csv):
    # Initialize an empty list to store image data and metadata
    data = []

    # Loop through all images in the folder
    for idx, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith(".png"):
            filepath = os.path.join(image_folder, filename)
            # Read the image
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (128, 128))
            image = image / 255.
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
            image = np.uint8(image * 255.)
            # Flatten the image into a 1D array
            image_flat = image.flatten()
            # Add ID, ImageID (filename), and pixel values
            row = [idx, filename] + image_flat.tolist()
            data.append(row)
    
    # Create a DataFrame
    num_columns = len(data[0]) - 2 if data else 0
    column_names = ["id", "ImageID"] + [indx for indx in range(num_columns)]
    df = pd.DataFrame(data, columns=column_names)

    # Save to CSV
    df.to_csv(output_csv, index=False)

# Paths for prediction and ground truth images
predictions_folder = "data/sample_solution"

# Output CSV paths
predictions_csv = "predictions.csv"

# Convert prediction images to CSV
images_to_csv_with_metadata(predictions_folder, predictions_csv)
