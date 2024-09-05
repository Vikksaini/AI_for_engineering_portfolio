import os
import random
import shutil
import labelme2coco


def convert_labelme_to_coco(labelme_folder, output_json_path):
    """
    Convert LabelMe annotations to COCO format.
    """
    # Convert Labelme annotations to COCO format
    labelme2coco.convert(labelme_folder, output_json_path)
    print(f"Converted Labelme annotations in {labelme_folder} to COCO format at {output_json_path}")


# Paths
labelme_folder = "log-labelled"  # Replace with your Labelme annotation folder path
output_json_path = "log_annotations_coco.json"  # Replace with your desired output path

# Convert annotations
convert_labelme_to_coco(labelme_folder, output_json_path)


# Split dataset into training and test sets
def split_dataset(image_dir, output_dir, test_ratio=0.1):
    """
    Split dataset into training and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = os.listdir(image_dir)
    random.shuffle(images)

    # Define test set size
    test_size = int(len(images) * test_ratio)
    test_images = images[:test_size]
    train_images = images[test_size:]

    # Move images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(test_dir, img))

    print(f"Split dataset into {len(train_images)} training and {len(test_images)} test images.")


# Call the function to split the dataset
image_dir = "log-labelled"  # Replace with your dataset path
output_dir = "split-dataset"
split_dataset(image_dir, output_dir, test_ratio=0.1)
