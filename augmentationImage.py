import os
import random
from PIL import Image
from torchvision.transforms import functional as F

# Directory containing the images
image_dir = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/val/Welding_Defects'

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Augmentation parameters
flip_horizontal_prob = 0.5  # Probability of horizontal flip
flip_vertical_prob = 0.5    # Probability of vertical flip
rotate_prob = 0.5           # Probability of rotation
rotate_angles = [90, -90, 180]  # Rotation angles

# Iterate over each image file
for filename in image_files:
    # Open the image
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)
    
    # Apply random horizontal flip
    if random.random() < flip_horizontal_prob:
        image = F.hflip(image)
    
    # Apply random vertical flip
    if random.random() < flip_vertical_prob:
        image = F.vflip(image)
    
    # Apply random rotation
    if random.random() < rotate_prob:
        angle = random.choice(rotate_angles)
        image = F.rotate(image, angle)
    
    # Save the augmented image
    augmented_image_path = os.path.join(image_dir, f'Augmented_' + filename)
    image.save(augmented_image_path)
    
    print(f"Augmented {filename} saved as augmented_{filename}")