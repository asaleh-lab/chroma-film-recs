"""Meaning search plus metadata filters, then one film for tonight."""

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
    return f"{film['title']}. {film['description']}"


def show(heading, results):
    print(f"\n{heading}")
    for film_id, document, distance, meta in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        print(
            f"{distance:.4f}  {film_id}  {meta['language']}  "
            f"{meta['minutes']} min  this week={meta['playing_this_week']}"
        )
        print(f"  {document}")


embed = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

# Start clean. The update/delete script drops this collection.
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
print(f"Query: {QUERY}")

show("Meaning only", collection.query(query_texts=[QUERY], n_results=3))

# Same query. Only films on this week, under 110 minutes, in English.
where = {
    "$and": [
        {"playing_this_week": {"$eq": True}},
        {"minutes": {"$lt": 110}},
        {"language": "en"},
    ]
}
filtered = collection.query(query_texts=[QUERY], n_results=2, where=where)
show("This week, under 110 minutes, English", filtered)

top = filtered["metadatas"][0][0]
print(f"\nTonight we put on: {top['title']}")
