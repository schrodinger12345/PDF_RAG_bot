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

    def upsert(self, ids , vectors, payloads):
        point = [PointStruct(id=ids[i], vector=vectors[i], payload = payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=point)

    def search(self, query_vector, top_k: int = 5):
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )
        context = []
        sources = set()
        for r in res:
            payload = getattr(r, "payload",None) or {}
            text = payload.get("text","")
            source = payload.get("source","")
            if text:
                context.append(text)
                sources.add(source)
        return {"context": "\n".join(context), "sources": list(sources)}
    
