# chroma-film-recs

This repo demonstrates how to keep a small catalog as vectors and search it by meaning. The example we will use is this week's films at The Lantern, a one-screen cinema. In our case we need a film for a tired evening, even when the text never says the word we typed.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Let's score two films by hand

```powershell
python src/cosine_by_hand.py
```

## Now we put the catalog in Chroma

```powershell
python src/create_collection.py
```
