"""Retrieved films stuffed into a prompt, then an answer from those hits."""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # OPENAI_API_KEY and OPENAI_MODEL from .env

QUERY = "something quiet after a long day"
COLLECTION = "lantern_films"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

films = json.loads(Path("data/films.json").read_text(encoding="utf-8"))


def document_for(film):
    return f"{film['title']}. {film['description']}"


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

# Same filter as filter_recommend.py. Only films we can put on tonight.
where = {
    "$and": [
        {"playing_this_week": {"$eq": True}},
        {"minutes": {"$lt": 110}},
        {"language": "en"},
    ]
}
hits = collection.query(query_texts=[QUERY], n_results=2, where=where)

print(f"Query: {QUERY}")
print("Retrieved:")
for document, meta in zip(hits["documents"][0], hits["metadatas"][0]):
    print(f"- {meta['title']} ({meta['minutes']} min)")
    print(f"  {document}")

# The model never sees the rest of the catalog. Only these documents go in the prompt.
context = "\n\n".join(hits["documents"][0])
system = (
    "Answer from these films only. If none of them fit the need, say so. "
    "Name the film you would put on tonight."
)
user = f"Films:\n{context}\n\nNeed: {QUERY}"

print("\nFilled prompt:")
print(f"system: {system}")
print(f"user: {user}")
print("---")

chat = OpenAI()
completion = chat.chat.completions.create(
    model=MODEL,
    temperature=0,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
)
print(completion.choices[0].message.content)
