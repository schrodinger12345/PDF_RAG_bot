from fastapi import FastAPI
import pydantic
import logging
import inngest
import inngest.fast_api
import uuid
import os
from dotenv import load_dotenv
import datetime
from inngest.experimental import ai
from data_loader import load_and_chunk_pdfs, get_embeddings
from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="RAGpdfApp",
    logger=logging.getLogger("uvicorn"),
    is_production= False,
    serializer= inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id= "RAG : Ingest PDF",
    trigger= inngest.TriggerEvent(event="rag/ingest-pdf")
)
async def ingest_pdf(ctx:inngest.Context):
    return {"status": "Api is working!"}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client,[ingest_pdf])