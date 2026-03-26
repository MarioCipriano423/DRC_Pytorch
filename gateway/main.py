# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingRespose
import httpx
import io

from src.dl_pipeline_interface import DLPipelineInterface

app = FastAPI()
dl_pipeline = DLPipelineInterface()

origins = [
    "*"
]

origins.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run-dl-pipeline")
async def run_dl_pipeline():
    result = await dl_pipeline.run_dl_pipeline()
    
    image_url = result["internal_plot_url"]

    async with httpx.AsyncClient() as client:
        img_response = await client.get(image_url)

    return StreamingRespose(
        io.BytesIO(img_response.content),
        media_type="image/png"
    )
