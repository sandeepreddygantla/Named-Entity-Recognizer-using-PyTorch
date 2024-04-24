import sys

from datasets import load_dataset

from ner.config.configuration import Configuration
from ner.entity.config_entity import DataIngestionConfig
from ner.exception import CustomException
from ner.logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        logger.info(" Data Ingestion component started...")
        self.data_ingestion_config = data_ingestion_config

    def get_data(self):
        try:
            logger.info("Loading data from HuggingFace...")
            pan_en_data = load_dataset(
                self.data_ingestion_config.dataset_name,
                name=self.data_ingestion_config.subset_name,
            )
            logger.info(f"Data Info : {pan_en_data}")

            return pan_en_data
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    project_config = Configuration()
    ingestion = DataIngestion(project_config.get_data_ingestion_config())
