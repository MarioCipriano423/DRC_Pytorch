# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingRespose
import httpx

app = FastAPI()

origins = [
    "*"
]

origins = [
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
]
