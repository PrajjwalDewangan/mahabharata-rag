import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

COLLECTION_NAME = "mahabharata"
BATCH_SIZE = 64
MODEL_NAME = "all-mpnet-base-v2"

def embed_and_store(chunks_path: str = "data/chunks.json",
                    chroma_path: str = "data/chroma"):
    
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, encoding = "utf-8") as f:
        chunks = json.load(f)
    print(f"Total chunks to embed: {len(chunks)}")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_path)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(
        name = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )

    print("Embedding and storing chunks...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [{
            "parva_number": c["parva_number"],
            "parva_name": c["parva_name"],
            "section_number": c["section_number"],
            "section_heading": c["section_heading"],
            "chunk_index": c["chunk_index"]
        } for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids = ids,
            documents = texts,
            embeddings = embeddings,
            metadatas = metadatas
        )
    
    print(f"\nDone. {collection.count()} chunks stored in CHromaDB.")
    print(f"ChromaDB persisted at: {chroma_path}")

if __name__ == '__main__':
    embed_and_store()

    