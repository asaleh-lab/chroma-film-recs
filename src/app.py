"""The Lantern recommender behind a Gradio chat."""

import json
import os
from pathlib import Path

import chromadb
import gradio as gr
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # OPENAI_API_KEY and OPENAI_MODEL from .env

COLLECTION = "lantern_films"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM = (
    "Answer from these films only. If none of them fit the need, say so. "
    "Name the film you would put on tonight."
)

films = json.loads(Path("data/films.json").read_text(encoding="utf-8"))


def document_for(film):
    return f"{film['title']}. {film['description']}"


embed = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

# Same catalog as the scripts. Built once when the app starts.
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

openai_client = OpenAI()


def chat(message, _history):
    hits = collection.query(query_texts=[message], n_results=2, where=where)
    documents = hits["documents"][0]
    metadatas = hits["metadatas"][0]
    if not documents:
        return "None of the films we can show tonight fit that."

    context = "\n\n".join(documents)
    completion = openai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Films:\n{context}\n\nNeed: {message}"},
        ],
    )
    titles = ", ".join(meta["title"] for meta in metadatas)
    return f"{completion.choices[0].message.content}\n\nHits: {titles}"


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        type="messages",
        title="The Lantern",
        examples=[
            "something quiet after a long day",
            "something loud",
        ],
    ).launch()
