import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_app.vector_store import ChromaVectorStore

store = ChromaVectorStore()
print('Client created, collection name:', store.COLLECTION_NAME)
# perform a raw query via the chroma client to inspect returned keys
results = store.collection.query(query_texts=['test query'], n_results=1, include=['documents','metadatas','distances','embeddings','uris','data'])
print('Result keys:', list(results.keys()))
for k, v in results.items():
    print(k, type(v))
    try:
        print('Sample:', v)
    except Exception as e:
        print('Could not print value for', k, e)
