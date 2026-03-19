# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from abc import ABC, abstractmethod

class BaseDLPipeline(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def execute_dl_pipeline(self):
        pass
