import copy
import os
import sys
import cv2
import torch
import numpy as np
import detectron2.data.transforms as T
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from detectron2.data import DatasetCatalog, MetadataCatalog, detection_utils as utils


class DataTransformation:
    def __init__(self, dataset_dict):
        try:
            logging.info(f"Initializing DataTransformation for {dataset_dict.get('file_name', 'unknown file')}")
            self.dataset_dict = copy.deepcopy(dataset_dict)

            # Validate input
            if "file_name" not in self.dataset_dict:
                raise ValueError("Dataset dictionary must contain 'file_name' key")

            # Read image with error handling
            try:
                self.image = utils.read_image(self.dataset_dict["file_name"], format="BGR")
            except Exception as e:
                logging.error(f"Error reading image {self.dataset_dict['file_name']}: {e}")
                raise CustomException(e,sys)

            logging.info("DataTransformation initialization successful")

        except Exception as e:
            logging.error(f"Error in DataTransformation initialization: {e}")
            raise CustomException(e,sys)

    def transform_data(self):
        try:
            logging.info("Starting data transformation")

            # Define transformation list with validated parameters
            transform_list = [
                T.Resize((800, 600)),
                T.RandomBrightness(0.8, 1.8),
                T.RandomContrast(0.6, 1.3),
                T.RandomSaturation(0.8, 1.4),
                T.RandomRotation(angle=[90, 90]),
                T.RandomLighting(0.7),
                T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
            ]

            # Apply transformations
            try:
                preprocessed_image, transforms = T.apply_transform_gens(transform_list, self.image)
            except Exception as e:
                logging.error(f"Error applying transformations: {e}")
                raise CustomException(e,sys)

            # Convert preprocessed image to tensor
            self.dataset_dict["image"] = torch.as_tensor(
                preprocessed_image.transpose(2, 0, 1).astype("float32")
            )

            # Validate annotations
            if "annotations" not in self.dataset_dict:
                logging.warning("No annotations found in dataset dictionary")
                return self.dataset_dict

            # Transform annotations
            try:
                annos = [
                    utils.transform_instance_annotations(obj, transforms, preprocessed_image.shape[:2])
                    for obj in self.dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

                # Create instances
                instances = utils.annotations_to_instances(annos, preprocessed_image.shape[:2])
                self.dataset_dict["instances"] = utils.filter_empty_instances(instances)

            except Exception as e:
                logging.error(f"Error processing annotations: {e}")
                raise CustomException(e,sys)

            logging.info("Data transformation completed successfully")
            return self.dataset_dict

        except Exception as e:
            logging.error(f"Unexpected error in transform_data: {e}")
            raise CustomException(e,sys)