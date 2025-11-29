import os
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model import Model
import dagshub
dagshub.init(repo_owner='rishendra-manne', repo_name='shrimp_seeds_detection', mlflow=True)


def main():
    data_ingestion=DataIngestion(0.7)
    data_preprocessing=DataPreprocessing()
    model=Model()
    data_ingestion.get_data("https://universe.roboflow.com/ds/OoffgDRtBF?key=nHXCifHwcG")
    data_ingestion.extract_data()
    train_images= data_ingestion.load_images_as_numpy(os.path.join("src","artifacts","dataset","train"))
    valid_images=data_ingestion.load_images_as_numpy(os.path.join("src","artifacts","dataset","valid"))
    test_images=data_ingestion.load_images_as_numpy(os.path.join("src","artifacts","dataset","test"))

    train_path=data_preprocessing.preprocess_images(train_images)
    valid_path=data_preprocessing.preprocess_images(valid_images)
    test_path=data_preprocessing.preprocess_images(test_images)
    data_ingestion.register_instances(train_path,test_path,valid_path)
    model.train_model()


main()
