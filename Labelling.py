import os
from PIL import Image
import cv2 as cv
import os
import numpy as np

# Define input and output directories, This was chnage to binary, ostu and k-clustering depenidng on where the annotated images are stored from masking

input_dir = 'Dataset/otsu/otsutest/'
output_dir = 'Dataset/test/labels-ostu'

#input_dir = '../Dataset/otsu/otsuvalid/'
#output_dir = '../Dataset/valid/labels-ostu'

#input_dir = '../Dataset/otsu/otsuvalid/'
#output_dir = '../Dataset/train/labels-otsu'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the input directory
for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    
    # Load the binary mask and get its contours
    mask = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # Write the polygons to a text file in the output directory
    output_file = os.path.join(output_dir, j[:-4] + '.txt')
    with open(output_file, 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))