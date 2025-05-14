import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def images_to_csv(image_folder, csv_file):
    # List to store image data
    image_data = []

    # Iterate over each image in the folder
    for image_name in tqdm(os.listdir(image_folder)):
        if image_name.endswith('.png') or image_name.endswith('.jpg'):
            # Load the image as a numpy array
            image_path = os.path.join(image_folder, image_name)
            img = Image.open(image_path)
            img_array = np.array(img)
            # Flatten the image array and convert it to a list
            img_flat = img_array.flatten().tolist()

            # Append the image name and flattened pixels to the image_data list
            image_data.append([image_name] + img_flat)

    # Create a DataFrame with the image data
    df = pd.DataFrame(image_data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False, header=False)

def csv_to_images(csv_file, output_folder):
    # Load the CSV file
    df = pd.read_csv(csv_file, header=None)

    # Iterate over each row in the CSV file
    for index, row in tqdm(df.iterrows()):
        # The first column is the image name
        image_name = row[0]

        # The rest of the columns are the pixel values, reshape them to the original image shape
        img_flat = row[1:].values.astype(np.uint8)  # Assuming the images are 8-bit depth
        img_size = int(np.sqrt(len(img_flat)))  # Assuming square images
        img_array = img_flat.reshape((img_size, img_size))

        # Convert the numpy array back to an image and save it
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_folder, image_name))

# Example usage
image_folder = 'data/sample_solution'
csv_file = 'data/depth_images.csv'
output_folder = 'data/output_images'
os.makedirs(output_folder, exist_ok=True)
# Convert images to CSV
images_to_csv(image_folder, csv_file)


# Convert CSV back to images
# csv_to_images(csv_file, output_folder)

# import zipfile

# def compress_images_to_zip(image_folder, zip_file):
#     with zipfile.ZipFile(zip_file, 'w') as zipf:
#         for image_name in os.listdir(image_folder):
#             if image_name.endswith('.png') or image_name.endswith('.jpg'):
#                 # Compress the image and add it to the ZIP file
#                 image_path = os.path.join(image_folder, image_name)
#                 zipf.write(image_path, arcname=image_name)

# # Example usage:
# image_folder = '/home/vinayak/DLP-Kaggle/data/testing_depths_solution'
# zip_file = 'compressed_images.zip'
# compress_images_to_zip(image_folder, zip_file)