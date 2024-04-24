import sys
from typing import Dict, List

import pandas as pd

from ner.entity.config_entity import DataValidationConfig
from ner.exception import CustomException
from ner.logger import logger


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data: Dict):
        logger.info("DataValidation component started...")
        self.data_validation_config = data_validation_config
        self.data = data

    def check_columns_names(self) -> bool:
        try:
            logger.info(" Checking Columns of all the splits ")
            column_check_result = list()

            for split_name in self.data_validation_config.data_split:
                column_check_result.append(
                    sum(
                        pd.DataFrame(self.data[split_name]).columns
                        == self.data_validation_config.columns_check
                    )
                )

            logger.info(f" Check Results {column_check_result}")

            if sum(column_check_result) == len(
                self.data_validation_config.data_split
            ) * len(self.data_validation_config.columns_check):
                return True
            else:
                return False

        except Exception as e:
            raise CustomException(e, sys)

    def type_check(self) -> bool:
        try:
            """Implement type checking as an assignment"""
            logger.info(f" Checking datatypes of all columns")
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def null_check(self) -> bool:
        try:
            """Implement null checking as an assignment"""
            logger.info(f" Checking nulls of all columns")
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def drive_checks(self) -> List[List[bool]]:
        logger.info(f" Checks initiated")
        checks = list()
        checks.append(
            [self.check_columns_names(), self.type_check(), self.null_check()]
        )
        return checks
