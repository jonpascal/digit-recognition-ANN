import struct
import numpy as np

def read_idx_ubyte(images_file, labels_file):
    '''Reads the idx files and outputs a list of tuples with pixel grayness values 
    as a np array for the first element and the label of the image for the second 
    element.'''

    images = []
    labels = []

    # Oben with rb as data should be read as a sequence of bytes rather than txt
    with open(images_file, 'rb') as f:
        # Read the magic number (first 4 bytes - uint)
        magic_number = struct.unpack('>I', f.read(4))[0]

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

    # Read the labels associated with each image
    with open(labels_file, 'rb') as f:
        # Read the magic number (first 4 bytes - uint)
        magic_number = struct.unpack('>I', f.read(4))[0]

        # Read the number of labels (next 4 bytes - uint)
        num_labels = struct.unpack('>I', f.read(4))[0]

        # Read the labels as bytes
        labels_data = struct.unpack(f'>{num_labels}B', f.read())

    for elt in labels_data: 
        labels.append(np.array([0 if elt != i else 1 for i in range(10)]))

    # Iterate through the images and convert them to 784-long vectors and create tuples
    image_label_tuples = zip(images, labels)

    return image_label_tuples


