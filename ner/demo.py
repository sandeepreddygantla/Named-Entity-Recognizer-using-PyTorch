from ner.components.data_ingestion import DataIngestion
from ner.config.configuration import Configuration
from ner.components.data_validation import DataValidation

project_config = Configuration()
ingestion = DataIngestion(project_config.get_data_ingestion_config())
en_data = ingestion.get_data()

validate = DataValidation(
    data_validation_config=project_config.get_data_validation_config(), data=en_data
)
check = validate.drive_checks()

def validate_check():
    for i in check[0]:
        if i == False:
            return False
        else:
            return True
