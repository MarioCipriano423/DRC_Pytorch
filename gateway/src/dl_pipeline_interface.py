# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from src.modules.dl_pipeline_core import DLPipeline

class DLPipelineInterface:
    def __init__(slf):
        pass

    def run_dl_pipeline(self, *args, **kwargs):
        dl_pipeline = DLPipeline()
        dl_pipeline.setup()
        return dl_pipeline.execute_dl_pipeline(*args, **kwargs)
