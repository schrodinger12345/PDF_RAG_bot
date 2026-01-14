from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance , PointStruct

class QdrantStorage:
    def __init__(self, urls="http://localhost:6333",collection= "documents",dim = 3072):
        self.client = QdrantClient(url=urls, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )