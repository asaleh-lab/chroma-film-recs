"""Update one film, delete one film, list and drop the collection."""

import os
import sys

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY from .env

COLLECTION = "lantern_films"
UPDATE_ID = "lost-in-translation"
DELETE_ID = "independence-day"

# Same model as create_collection.py. Needed so a later query would still embed.
embed = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

client = chromadb.PersistentClient(path="chroma_data")
if COLLECTION not in [c.name for c in client.list_collections()]:
    print("No lantern_films collection. Run: python src/create_collection.py")
    sys.exit(1)

collection = client.get_collection(name=COLLECTION, embedding_function=embed)
print(f"Films in the collection: {collection.count()}")

before = collection.get(ids=[UPDATE_ID])
print(f"\nBefore update: {before['documents'][0]}")

# The box office reprints the card. Chroma re-embeds the new text.
collection.update(
    ids=[UPDATE_ID],
    documents=[
        "Lost in Translation. Two jet-lagged strangers. Quiet hotel corridors, a karaoke room, almost no plot."
    ],
)

after = collection.get(ids=[UPDATE_ID])
print(f"After update: {after['documents'][0]}")

collection.delete(ids=[DELETE_ID])
print(f"\nTook down {DELETE_ID}. Films left: {collection.count()}")

print("\nCollections:")
print([c.name for c in client.list_collections()])

client.delete_collection(COLLECTION)
print("Dropped lantern_films.")
print([c.name for c in client.list_collections()])
