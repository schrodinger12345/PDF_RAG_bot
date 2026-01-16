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
from custom_types import RAGChunkAndSRC, RAGUpsertResponse, RAGSearchResponse, RAGQueryRes

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
    def _load(ctx: inngest.Context) -> RAGChunkAndSRC:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdfs(pdf_path)
        return RAGChunkAndSRC(
            source_id= source_id,
            pdf_chunks= chunks
        )

    def _upsert(ChunkAndSrc: RAGChunkAndSRC)-> RAGUpsertResponse:
        chunks = ChunkAndSrc.pdf_chunks
        source_id = ChunkAndSrc.source_id
        vecs = get_embeddings([chunk for chunk in chunks])
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        payloads = [
            {
                "source_id": source_id,
                "text": chunk,
            }
            for chunk in chunks
        ]
        qdrrant = QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResponse(ingested=len(ids))

    chunk_and_src = await ctx.step.run("Loading and chunking PDFs", lambda: _load(ctx=ctx), output_type=RAGChunkAndSRC)
    inngested = await ctx.step.run("Embedding and upserting", lambda: _upsert(ChunkAndSrc=chunk_and_src), output_type=RAGUpsertResponse)
    return inngested.model_dump()

@inngest_client.fast_api.create_function(
    fn_id="RAG : Search PDF",
    trigger=inngest.TriggerEvent(event="rag/search-pdf")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str , top_k: int =5) -> RAGSearchResponse:
        query_vec = get_embeddings([question])[0]
        store = QdrantStorage()
        res = store.search(query_vector=query_vec, top_k=top_k)
        return RAGSearchResponse(context=res["context"], sources=res["sources"])


app = FastAPI()

inngest.fast_api.serve(app, inngest_client,[ingest_pdf])