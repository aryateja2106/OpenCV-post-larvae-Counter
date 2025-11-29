import os
import sys
import cv2
import shutil
import json
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np


@dataclass
class PreprocessingConfig:
    preprocessed_train_path: str = os.path.join(
        os.getcwd(),'src','artifacts',
        'preprocessed',
        'train'
    )
    os.makedirs(preprocessed_train_path, exist_ok=True)


class DataPreprocessing:
    def __init__(self):
        self.preprocessing_config = PreprocessingConfig()
        # Create directory if it doesn't exist
        os.makedirs(self.preprocessing_config.preprocessed_train_path, exist_ok=True)

    def apply_clahe_color(self, image):
        """
        Apply CLAHE to each color channel while preserving color information

        Args:
            image: Input BGR image

        Returns:
            processed_img: Color image after CLAHE application
        """
        try:
            # Split the image into channels
            b, g, r = cv2.split(image)

            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # Apply CLAHE to each channel
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)

            # Merge the channels back
            processed_img = cv2.merge([b_clahe, g_clahe, r_clahe])

            return processed_img
        except Exception as e:
            logging.error(f"Error in apply_clahe_color: {str(e)}")
            raise CustomException(e, sys)

    def preprocess_images(self, images: dict):
        """
        Preprocess multiple images with color-preserving CLAHE and Gaussian blur

        Args:
            images: Dictionary with filename as key and image array as value

        Returns:
            str: Path to the directory containing preprocessed images
        """
        try:
            for filename, img in images.items():
                save_path = os.path.join(
                    self.preprocessing_config.preprocessed_train_path,
                    filename
                )

                # Apply color-preserving preprocessing
                clahe_image = self.apply_clahe_color(img)
                blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)

                # Save the preprocessed image
                cv2.imwrite(save_path, blurred_image)

            return self.preprocessing_config.preprocessed_train_path

        except Exception as e:
            logging.error(f"Error in preprocess_images: {str(e)}")
            raise CustomException(e, sys)
