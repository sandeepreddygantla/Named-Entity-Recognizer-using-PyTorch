import os
import sys

from ner.constants import *
from ner.entity.config_entity import (
    DataIngestionConfig,
)
from ner.exception import CustomException
from ner.logger import logger
from ner.utils import read_config


class Configuration:
    def __init__(self):
        try:
            logger.info("Reading the configuration file...")
            self.config = read_config(file_name=CONFIG_FILE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            dataset_name = self.config[DATA_INGESTION_KEY][DATASET_NAME]
            subset_name = self.config[DATA_INGESTION_KEY][SUBSET_NAME]
            data_store = os.path.join(
                os.getcwd(),
                self.config[PATH_KEY][ARTIFACTS_KEY],
                self.config[PATH_KEY][DATA_STORE_KEY],
            )

            data_ingestion_config = DataIngestionConfig(
                dataset_name=dataset_name, subset_name=subset_name, data_path=data_store
            )
            return data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)