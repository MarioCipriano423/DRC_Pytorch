# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from src.base_dl_pipeline import BaseDLPipeline
import os
import htppx
from dotenv import load_dotenv

load_dotenv()

class DLPipeline(BaseDLPipeline):

    #URLs

    def __init__(self):
        pass

    def setup(self):
        pass

    async def execute_dl_pipeline(self, file):

        async with htppx.AsynClient() as client:

            #DEFINE DL PIPELINE
            return "Work in progress..."

