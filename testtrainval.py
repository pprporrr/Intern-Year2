import os
import random
from sklearn.model_selection import train_test_split
from shutil import copyfile

# Path to the original dataset directory
dataset_dir = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/OriginalResize/Welding/Welding_Defects(LQ-WD)'

# Output directory for the split datasets
output_dir = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/DatasetsTestTrainVal/test/Welding/Welding_Defects(LQ-WD)'
3
# Random seed for reproducibility
random_seed = 42

# Split ratios (train:val:test)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# Get the list of image files in the dataset directory
image_files = [filename for filename in os.listdir(dataset_dir) if filename.endswith('.jpg')]

# Split the dataset into train, validation, and test sets
train_files, testval_files = train_test_split(image_files, train_size=train_ratio, random_state=random_seed)
val_files, test_files = train_test_split(testval_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=random_seed)

# Copy train files to output directory
for filename in train_files:
    src = os.path.join(dataset_dir, filename)
    dst = os.path.join(output_dir, 'train', filename)
    copyfile(src, dst)

# Copy validation files to output directory
for filename in val_files:
    src = os.path.join(dataset_dir, filename)
    dst = os.path.join(output_dir, 'val', filename)
    copyfile(src, dst)

# Copy test files to output directory
for filename in test_files:
    src = os.path.join(dataset_dir, filename)
    dst = os.path.join(output_dir, 'test', filename)
    copyfile(src, dst)
