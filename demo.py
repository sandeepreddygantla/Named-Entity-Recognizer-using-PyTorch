from ner.components.data_ingestion import DataIngestion
from ner.config.configuration import Configuration
from ner.components.data_validation import DataValidation
from ner.components.data_prepration import DataPreprocessing
from ner.components.model_trainer import TrainTokenClassifier

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

def validate_check():
    for i in check[0]:
        if i == False:
            return False
        else:
            return True


if validate_check() == True:
    processed_data = DataPreprocessing(
        data_preprocessing_config=project_config.get_data_preprocessing_config(),
        data=en_data,
    )
    p_data = processed_data.prepare_data_for_fine_tuning()
    print(p_data)
    classifer = TrainTokenClassifier(
        model_trainer_config=project_config.get_model_trainer_config(),
        processed_data=p_data,
    )
    classifer.train()