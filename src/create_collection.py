"""Create a Chroma collection, add the films, query by meaning."""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY from .env

QUERY = "something quiet after a long day"
COLLECTION = "lantern_films"

films = json.loads(Path("data/films.json").read_text(encoding="utf-8"))


def document_for(film):
    # Chroma embeds this string, not the raw JSON row.
    return f"{film['title']}. {film['description']}"


# Same model as cosine_by_hand.py. Chroma calls it on add and on query.
embed = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

# Files under chroma_data/ so the collection survives this process.
client = chromadb.PersistentClient(path="chroma_data")
if COLLECTION in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION)  # start clean on reruns

collection = client.create_collection(
    name=COLLECTION,
    embedding_function=embed,
    metadata={"hnsw:space": "cosine"},  # same metric as the hand-scored script
)

collection.add(
    ids=[film["id"] for film in films],
    documents=[document_for(film) for film in films],
    metadatas=[
        {
            "title": film["title"],
            "language": film["language"],
            "minutes": film["minutes"],
            "playing_this_week": film["playing_this_week"],
        }
        for film in films
    ],
)

print(f"Stored {collection.count()} films")

# Embed the query, then nearest neighbours. Distance: lower is closer.
results = collection.query(query_texts=[QUERY], n_results=3)
print(f"Query: {QUERY}")
for film_id, document, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0],
):
    print(f"{distance:.4f}  {film_id}")
    print(f"  {document}")
