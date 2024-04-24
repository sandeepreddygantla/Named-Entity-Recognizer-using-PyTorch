import sys
from typing import Dict, List

from ner.components.data_ingestion import DataIngestion
from ner.components.data_prepration import DataPreprocessing
from ner.components.data_validation import DataValidation
from ner.components.model_trainer import TrainTokenClassifier
from ner.config.configuration import Configuration
from ner.exception import CustomException
from ner.logger import logger


class TrainPipeline:
    def __init__(self, config: Configuration):
        self.config = config

    def run_data_ingestion(
        self,
    ) -> Dict:
        try:
            logger.info("Running DataIngestion in TrainPipeline")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config()
            )
            data = data_ingestion.get_data()
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def run_data_validation(self, data) -> List[List[bool]]:
        try:
            logger.info("Running DataValidation in TrainPipeline")
            validation = DataValidation(
                data_validation_config=self.config.get_data_validation_config(),
                data=data,
            )
            checks = validation.drive_checks()
            return checks
        except Exception as e:
            raise CustomException(e, sys)

    def run_data_preparation(self, data) -> Dict:
        try:
            logger.info("Running the DataPreprocessing in TrainPipeline")
            data_preprocessing = DataPreprocessing(
                data_preprocessing_config=self.config.get_data_preprocessing_config(),
                data=data,
            )
            processed_data = data_preprocessing.prepare_data_for_fine_tuning()
            return processed_data
        except Exception as e:
            raise CustomException(e, sys)

    def run_model_training(self, processed_data):
        try:
            logger.info(" Running Model Training in TrainPipeline")
            classifier = TrainTokenClassifier(
                model_trainer_config=self.config.get_model_trainer_config(),
                processed_data=processed_data,
            )
            classifier.train()

            logger.info("Training Completed")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_check(check):
        for i in check[0]:
            if i == False:
                return False
            else:
                return True

    def run_pipeline(self):
        data = self.run_data_ingestion()
        checks = self.run_data_validation(data=data)
        if self.validate_check(checks):
            processed_data = self.run_data_preparation(data=data)
            logger.info("Processed data :", processed_data)
            self.run_model_training(processed_data=processed_data)
        else:
            logger.error("Checks Failed")
