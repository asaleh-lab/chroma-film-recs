"""Create a Chroma collection, add the films, query by meaning."""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

QUERY = "something quiet after a long day"
COLLECTION = "lantern_films"

films = json.loads(Path("data/films.json").read_text(encoding="utf-8"))


def document_for(film):
    return f"{film['title']}. {film['description']}"


embed = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

client = chromadb.PersistentClient(path="chroma_data")
if COLLECTION in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION)

collection = client.create_collection(
    name=COLLECTION,
    embedding_function=embed,
    metadata={"hnsw:space": "cosine"},
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

results = collection.query(query_texts=[QUERY], n_results=3)
print(f"Query: {QUERY}")
for film_id, document, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0],
):
    print(f"{distance:.4f}  {film_id}")
    print(f"  {document}")
