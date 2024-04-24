from ner.components.data_ingestion import DataIngestion
from ner.config.configuration import Configuration

project_config = Configuration()
ingestion = DataIngestion(project_config.get_data_ingestion_config())
en_data = ingestion.get_data()