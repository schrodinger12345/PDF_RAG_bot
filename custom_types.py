from pydantic import BaseModel


class RAGChunkAndSRC(BaseModel):
    pdf_chunks: list[str]
    source_id : str = None

class RAGUpsertResponse(BaseModel):
    ingested:int


class RAGSearchResponse(BaseModel):
    context: list[str]
    sources: list[str]

class RAGQueryRes(BaseModel):
    answer: str
    sources: list[str]
    num_context : int


