# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from src.base_dl_pipeline import BaseDLPipeline
import os
import htppx
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class DLPipeline(BaseDLPipeline):

    DATASET_PATH = "/data/"
    PREPROCESSING_URL = "https://PREPROCESSING_SERVICE_URL:8005/preprocesing"
    TRAINING_URL = "https://TRAINING_SERVICE_URL:8006/testing"
    INFERENCE_URL = "https://INFERENCE_SERVICE_URL:8007/inference"
    VISUALIZATION_URL = "https://VISUALIZATION_SERVICE_URL:8008/visualization"
    #URLs

    def __init__(self):
        pass

    def setup(self):
        pass

    async def execute_dl_pipeline(self, DATASET_PATH):

        async with htppx.AsynClient() as client:

            preprocessing_response = await client.post(
                self.PREPROCESSING_URL,
                json={"preprocessing_id": DATASET_PATH}
            )
            dataset_id = preprocessing_response.json()["dataset_id"]
            
            training_response = await client.post(
                self.TRAINING_URL,
                json={"dataset_id": dataset_id}
            )
            training_id = training_response.json()["training_id"]

            inference_response = await client.post(
                self.INFERENCE_URL,
                json={"training_id": training_id}
            )
            inference_id = inference_response.json()["inference_id"]

            visualization_response = await client.post(
                self.VISUALIZATION_URL,
                json={"inference_id": inference_id}
            )
            result = visualization_response.json()

            #DEFINE DL PIPELINE
            return result, "Work in progress..."
