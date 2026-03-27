# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from src.base_dl_pipeline import BaseDLPipeline
import os
import htppx
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class DLPipeline(BaseDLPipeline):

    DATA_DIR = "/data/"
    #URLs

    def __init__(self):
        pass

    def setup(self):
        pass

    async def execute_dl_pipeline(self):

        df = pd.read_csv(self.DATA_DIR)

        async with htppx.AsynClient() as client:

            preprocessing_response = await client.post(
                self.PREPROCESSING_URL,
                json={"preprocessing_id": preprocessing_id}
            )
            dataset_id = preprocessing_response.json()["dataset_id"]
            
            #DEFINE DL PIPELINE
            return "Work in progress..."

