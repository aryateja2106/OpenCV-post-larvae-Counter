import os
import sys
import mlflow
import torch
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from src.components.data_transformation import DataTransformation
from detectron2.data.datasets import register_coco_instances


@dataclass
class TrainingConfig:
    """Configuration class for model training parameters"""
    MLFLOW_TRACKING_URI: str = "https://dagshub.com/rishendra-manne/shrimp_seeds_detection.mlflow"
    MODEL_CONFIG_PATH: str = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    OUTPUT_DIR: str = "outputs"

    def get_model_config(self) -> CfgNode:
        """Creates and returns the model configuration"""
        try:
            

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.MODEL_CONFIG_PATH))
            cfg.DATASETS.TRAIN = ("my_dataset_train",)
            cfg.DATASETS.TEST = ("my_dataset_valid",)
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.MODEL_CONFIG_PATH)
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2048
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg.SOLVER.IMS_PER_BATCH = 2
            cfg.SOLVER.BASE_LR = 0.025
            cfg.SOLVER.MAX_ITER = 7000
            cfg.TEST.EVAL_PERIOD = 500
            cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
            cfg.SOLVER.STEPS = [1000,2000,3000,4000,5000,6000]
            cfg.SOLVER.GAMMA = 0.53
            cfg.OUTPUT_DIR = self.OUTPUT_DIR
            return cfg

        except Exception as e:
            logging.error(f"Error in creating model configuration: {e}")
            raise CustomException(e,sys)


class CustomTrainer(DefaultTrainer):
    """Custom trainer class extending DefaultTrainer"""

    @classmethod
    def build_train_loader(cls, cfg):
        """Build custom train loader with data transformation"""
        try:
            return build_detection_train_loader(
                cfg,
                mapper=lambda x: DataTransformation(x).transform_data()
            )
        except Exception as e:
            logging.error(f"Error building train loader: {e}")
            raise CustomException(e,sys)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator for model validation"""
        try:
            return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
        except Exception as e:
            logging.error(f"Error building evaluator: {e}")
            raise CustomException(e,sys)


class Model:
    def __init__(self):
        try:
            self.config = TrainingConfig()
            self.cfg = self.config.get_model_config()
            self.trainer = None
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            logging.info("Model initialization successful")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise CustomException(e, sys)

    def train_model(self):
        """Train the model with MLflow tracking"""
        try:
            logging.info("Starting model training")

            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("learning_rate", self.cfg.SOLVER.BASE_LR)
                mlflow.log_param("max_iterations", self.cfg.SOLVER.MAX_ITER)
                mlflow.log_param("batch_size", self.cfg.SOLVER.IMS_PER_BATCH)
                
                self.trainer = CustomTrainer(self.cfg)
                self.trainer.resume_or_load(resume=False)
                self.trainer.train()
                
                # Process metrics before logging
                latest_metrics = self.trainer.storage.latest()
                if latest_metrics:
                    processed_metrics = {}
                    for metric_name, metric_value in latest_metrics.items():
                        # Handle tuple metrics by taking the first value
                        if isinstance(metric_value, tuple):
                            processed_metrics[metric_name] = float(metric_value[0])
                        # Handle other numeric types
                        elif isinstance(metric_value, (int, float)):
                            processed_metrics[metric_name] = float(metric_value)
                        else:
                            logging.warning(f"Skipping metric {metric_name} with unsupported type {type(metric_value)}")
                    
                    # Log the processed metrics
                    if processed_metrics:
                        mlflow.log_metrics(processed_metrics)
                        logging.info(f"Logged metrics: {processed_metrics}")
                    else:
                        logging.warning("No valid metrics to log")
                
                self.save_model()
                logging.info("Model training completed successfully")

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)

    def save_model(self):
        """Save the trained model"""
        try:
            logging.info(f"Saving model to {self.cfg.OUTPUT_DIR}")
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            torch.save(self.trainer.model.state_dict(),
                      os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth"))
            with open(os.path.join(self.cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise CustomException(e, sys)
