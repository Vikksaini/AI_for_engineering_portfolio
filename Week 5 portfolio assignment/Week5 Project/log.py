import os
import sys
import numpy as np
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools.coco import COCO
import cv2
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask-RCNN-TF2")  # Replace with your Mask-RCNN-TF2 directory

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")  # Download this file if not present

class LogDetectionConfig(Config):
    """Configuration for training on the log detection dataset."""
    NAME = "log_detection"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + log class
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

config = LogDetectionConfig()

class LogsDataset(utils.Dataset):
    def load_logs(self, dataset_dir, subset):
        """
        Load a subset of the logs dataset.
        """
        self.add_class("logs", 1, "log")

        # Load annotations in COCO format
        coco = COCO(os.path.join(dataset_dir, f"log_annotations_coco.json"))
        image_ids = list(coco.imgs.keys())

        # Add images
        for i in image_ids:
            self.add_image(
                "logs",
                image_id=i,
                path=os.path.join(dataset_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i]))
            )

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        masks = np.zeros([image_info['height'], image_info['width'], len(annotations)], dtype=np.uint8)
        class_ids = np.array([1 for _ in range(len(annotations))], dtype=np.int32)  # All logs are class 1

        for i, annotation in enumerate(annotations):
            mask = coco.annToMask(annotation)
            masks[:, :, i] = mask
        
        return masks, class_ids

# Load datasets
dataset_train = LogsDataset()
dataset_train.load_logs("split-dataset/train", "train") 
dataset_train.prepare()

dataset_val = LogsDataset()
dataset_val.load_logs("split-dataset/test", "val") 

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Load pre-trained weights (COCO weights)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

class InferenceConfig(LogDetectionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Load trained weights
model_path = model.find_last()[1]
model.load_weights(model_path, by_name=True)



def detect_and_visualize(image_path, model):
    """
    Detect logs in an image and visualize results.
    """
    image = cv2.imread(image_path)
    results = model.detect([image], verbose=1)
    r = results[0]

    # Visualize results
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ["BG", "log"], r['scores'])

    return r

# Example usage
r = detect_and_visualize("split-dataset/test/sample_image.jpg", model)  

def count_detected_logs(r):
    """
    Count the number of detected logs in an image.
    """
    return len(r['rois'])

print(f"Number of detected logs: {count_detected_logs(r)}")
