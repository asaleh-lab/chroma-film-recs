# chroma-film-recs

This repo demonstrates how to keep a small catalog as vectors and search it by meaning. The example we will use is this week's films at The Lantern, a one-screen cinema. In our case we need a film for a tired evening, even when the text never says the word we typed.

**Article:** [Recommend a film by semantic search with Chroma and Gradio](https://wysiwygs.de/blog/recommend-a-film-by-semantic-search-chroma-gradio/)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Score two films by hand

```powershell
python src/cosine_by_hand.py
```

## Now we put the catalog in Chroma

```powershell
python src/create_collection.py
```

## Update a card and drop the collection

```powershell
python src/query_update_delete.py
```

## Filter the catalog and pick a film

```powershell
python src/filter_recommend.py
```

## Now we stuff the hits into a prompt

```powershell
python src/rag_from_hits.py
```

## Serve it with Gradio

```powershell
python src/app.py
```
