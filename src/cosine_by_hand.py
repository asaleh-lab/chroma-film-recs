"""Two embeddings, cosine scored without Chroma."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

QUERY = "something quiet after a long day"
films = {
    film["id"]: film
    for film in json.loads(Path("data/films.json").read_text(encoding="utf-8"))
}


def document_for(film):
    return f"{film['title']}. {film['description']}"


lost = document_for(films["lost-in-translation"])
independence = document_for(films["independence-day"])

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[QUERY, lost, independence],
)
query_vec, lost_vec, independence_vec = (item.embedding for item in response.data)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"Query: {QUERY}")
print(f"Lost in Translation: {cosine(query_vec, lost_vec):.4f}")
print(f"Independence Day: {cosine(query_vec, independence_vec):.4f}")
