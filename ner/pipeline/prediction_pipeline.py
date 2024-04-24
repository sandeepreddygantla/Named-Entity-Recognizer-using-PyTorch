import os
import sys

import torch
from transformers import AutoTokenizer

from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.config.configuration import Configuration
from ner.exception import CustomException


class PredictionPipeline:
    def __init__(self, config: Configuration):
        self.prediction_pipeline_config = config.get_model_predict_pipeline_config()
        self.tokenizer = self.prediction_pipeline_config.tokenizer
        self.tags = self.prediction_pipeline_config.tags

        if len(os.listdir(self.prediction_pipeline_config.output_dir)) == 0:
            raise LookupError(
                "Model not found: Please Run Model trainer before prediction pipeline"
            )
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.prediction_pipeline_config.output_dir
        )

    def predict(self, text):
        try:
            tokens = self.tokenizer(text).tokens()
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cpu")
            outputs = self.model(input_ids)[0]
            predictions = torch.argmax(outputs, dim=2)
            preds = [self.tags[p] for p in predictions[0].cpu().numpy()]
            # filtered_tags = []
            filtered_preds = []
            for token, pred in zip(tokens, preds):
                if "▁" in token[0]:
                    # filtered_tags.append(token)
                    filtered_preds.append(pred)
            return filtered_preds
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, data):
        prediction = self.predict(data)
        response = {"Input_data": data.split(), "Tags": prediction}
        return response
