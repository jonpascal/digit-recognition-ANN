import struct
import numpy as np
import os 
from PIL import Image, ImageOps

def read_idx_ubyte(images_file, labels_file):
    '''Reads the idx files and outputs a list of tuples with pixel grayness values 
    as a np array for the first element and the label of the image for the second 
    element.'''

    # Oben with rb as data should be read as a sequence of bytes rather than txt
    with open(images_file, 'rb') as f:
        # Read the magic number (first 4 bytes - uint)
        _ = struct.unpack('>I', f.read(4))[0]

        # Read the number of images (next 4 bytes - uint)
        num_images = struct.unpack('>I', f.read(4))[0]

        # Read the number of rows and columns in each image (next 2 * 4 bytes)
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        # Read the image data as bytes
        images_data = struct.unpack(f'>{num_images * num_rows * num_cols}B', f.read())

    # Convert the image data to a NumPy array
    images = list(images_data)
    images = [np.array(images[k*num_cols*num_rows : (k+1)*num_cols*num_rows]) for k in range(num_images)]
    images = [np.reshape(image, (784, 1)) for image in images]

    # Read the labels associated with each image
    with open(labels_file, 'rb') as f:
        # Read the magic number (first 4 bytes - uint)
        _ = struct.unpack('>I', f.read(4))[0]

        # Read the number of labels (next 4 bytes - uint)
        num_labels = struct.unpack('>I', f.read(4))[0]

        # Read the labels as bytes
        labels_data = struct.unpack(f'>{num_labels}B', f.read())

    labels = []
    for elt in labels_data: 
        labels.append(np.array([0 if elt != i else 1 for i in range(10)]))
    labels = [np.reshape(label, (10, 1)) for label in labels]

    # Iterate through the images and convert them to 784-long vectors and create tuples
    image_label_tuples = list(zip(images, labels))

    return image_label_tuples

def read_test_data(folder_name=None):
    '''Reads the images in the input_images folder and vectorises them.'''
    if folder_name is None: 
        folder_name = 'input_images'
    folder_path = os.path.join(os.getcwd(), folder_name)

    images = []

    output_folder = os.path.join(os.getcwd(), 'processed_images')
    os.makedirs(output_folder, exist_ok=True)

    # Go through all the files in the folder 
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')):
            with Image.open(file_path) as img: 

                # Make the image 28x28 pixels and grayscale. We invert the picture 
                # as it is usually written on white paper with a black marker 
                img = img.convert('L')
                img = img.resize((28, 28))
                img = ImageOps.invert(img)

                pixel_values = [int(value) for value in img.getdata()]
                images.append(np.array(pixel_values))
                images = [np.resize(image, (784, 1)) for image in images]

                output_path = os.path.join(output_folder, file_name)
                img.save(output_path)

    return images

