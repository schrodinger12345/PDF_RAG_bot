from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-pro-embedding-002"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdfs(path:str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for text in texts:
        split_texts = splitter.split_text(text)
        chunks.extend(split_texts)
    return chunks

def get_embeddings(texts:list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model = EMBED_MODEL,
        input = texts
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings