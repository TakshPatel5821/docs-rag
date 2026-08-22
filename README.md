# docs-rag

Retrieval-augmented question answering over a corpus of **SEC 10-K filings**, in
Python. Ask a question in plain English; the service embeds it, finds the
nearest passages by **cosine similarity** in PostgreSQL/pgvector, and returns a
generated answer **with the filings it came from**.

The corpus is four real annual reports pulled from EDGAR — Apple, Microsoft,
JPMorgan Chase and Bank of America — about 2.6 MB of text and
**4,814 chunks** at the default settings.

```
POST /query {"question": "How does JPMorgan Chase determine its allowance for credit losses?"}
   |
   v
embed query (all-MiniLM-L6-v2, 384-d)  ->  SELECT ... ORDER BY embedding <=> $1 LIMIT 5
   |                                                      (pgvector, IVFFlat, cosine)
   v
prompt = system grounding rule + numbered passages + question
   |
   v
LLMProvider.generate()  ->  Anthropic | OpenAI | Ollama | echo   ->  {answer, sources[]}
```

---

## Quick start

Three commands from a clean clone:

```bash
docker compose up -d --build
docker compose exec app python -m app.cli ingest data/filings
docker compose exec app python -m app.cli build-index
```

Then ask it something:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What does Apple say about its reliance on single or limited source suppliers?"}'
```

Interactive OpenAPI docs: <http://localhost:8000/docs>. Health and corpus size:
`curl localhost:8000/health`.

The corpus is committed, so nothing needs downloading. To rebuild it from EDGAR
(stdlib only, no dependencies required):

```bash
SEC_USER_AGENT="your-name your-email" python scripts/fetch_filings.py --tickers AAPL MSFT JPM BAC
```

### Generation providers

Retrieval works with no credentials at all. Generation is provider-agnostic and
chosen at runtime with `LLM_PROVIDER`:

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `echo` (default) | nothing | Returns the retrieved context instead of a generated answer. Lets the whole pipeline — and the test suite — run with zero credentials. |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5` by default, via the official `anthropic` SDK. |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` by default. |
| `ollama` | a local Ollama | Self-hosted; `OLLAMA_BASE_URL`, `OLLAMA_MODEL`. |

Copy `.env.example` to `.env` and edit. Adding a fifth provider means one class
with one `generate()` method and one dict entry — nothing else in the codebase
imports a vendor SDK.

---

## Worked example

<!--WORKED_EXAMPLE-->

---

## How it works

### 1. Ingestion — `app/ingest.py`, `app/chunking.py`

Filings are extracted to text (`.txt`/`.md` directly, `.pdf` through `pypdf`)
and split into **800-character chunks with 150 characters of overlap**,
preferring paragraph boundaries, then sentence boundaries, then whitespace. The
overlap prefix is counted *inside* the 800-character budget, so no chunk ever
exceeds the limit.

**Why 800/150.** 800 characters (~150–200 tokens) is large enough to hold a
whole risk factor or accounting policy paragraph and small enough that the
embedding still points at one topic — a 4,000-character chunk covering five
subjects averages into a vector close to nothing in particular. 150 characters
(~19%) of overlap covers roughly one long filing sentence, so a fact that
straddles a boundary survives intact in at least one chunk. Those are the
reasons for the starting point; the numbers below are the reason for keeping it.

### 2. Embedding — `app/embeddings.py`

`sentence-transformers/all-MiniLM-L6-v2`, run locally: free, offline, no API
key, 384 dimensions, fast on CPU. Chunks are embedded in **batches of 32** —
one forward pass per batch instead of per chunk — and L2-normalised.

Everything downstream talks to an `Embedder` protocol, so swapping in OpenAI
`text-embedding-3-small` is a new class plus a config change. The protocol also
gives the tests a `HashEmbedder` double, which is why CI needs no model
download.

<!--INGEST_TIMING-->

### 3. Retrieval — `app/retrieval.py`

```sql
SELECT c.content, d.source, c.embedding <=> %s::vector AS distance
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %s::vector
LIMIT 5;
```

`<=>` is pgvector's **cosine distance** operator. Cosine rather than L2 because
MiniLM encodes meaning in the *direction* of the vector while magnitude largely
tracks how much text went in: under L2 a long passage and a short passage on the
same topic look far apart. (Vectors are normalised at write time, which makes
the two rank-equivalent — using the cosine operator keeps that true if an
un-normalised embedder is ever swapped in.)

**The IVFFlat index** is built *after* ingestion, by `app.cli build-index`, not
in the schema. IVFFlat is a trained index: it k-means the rows into `lists`
cells and probes only the nearest `ivfflat.probes` of them at query time, which
is what turns a sequential scan over every vector into approximate nearest
neighbour search. Building it against an empty table produces meaningless
centroids. `lists` is sized from the actual row count (pgvector's rule of thumb,
rows/1000). It is an *approximate* index — recall is traded for speed, tunable
per query with `ivfflat.probes`.

### 4. Generation — `app/prompt.py`, `app/generation/`

The system instruction is explicit: answer only from the provided passages, cite
them by number, and if the context does not contain the answer say exactly
*"The provided context does not contain the answer to that question."*
Passages are numbered `[1]…[5]` with their source filing and cosine similarity.

**When retrieval returns nothing, the model is never called at all** —
`app/service.py` short-circuits to the refusal. There is nothing to ground an
answer in, and a confident hallucination is the specific failure this design
exists to prevent.

Every response carries `sources[]`: filing, chunk index, cosine similarity and
an excerpt.

---

## Does retrieval actually work?

<!--EVAL-->

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

<!--TESTS-->

No test touches a real database or a real LLM: Postgres is replaced by a fake
connection that records the SQL it is handed, the model by `HashEmbedder`, and
generation by a recording double. The Anthropic path is covered with a mocked
client, and the pgvector round trip has its own module that skips itself unless
`DOCS_RAG_TEST_DSN` is set:

```bash
DOCS_RAG_TEST_DSN=postgresql://docsrag:docsrag@localhost:5432/docsrag pytest -m integration
```

---

## Layout

```
app/
  chunking.py     paragraph-aware splitting with overlap
  embeddings.py   Embedder protocol + MiniLM and hash implementations
  db.py           psycopg/pgvector access, IVFFlat index build
  ingest.py       read -> chunk -> batch embed -> persist
  retrieval.py    top-k cosine search
  prompt.py       grounding instruction + numbered context
  generation/     provider-agnostic LLM layer (anthropic|openai|ollama|echo)
  service.py      the RAG pipeline
  api.py          FastAPI: POST /query, GET /health
  cli.py          init-db | ingest | build-index | query | stats
eval/             10 questions with known answers + the recall harness
scripts/          EDGAR downloader (stdlib only)
sql/              schema
tests/            pytest suite
```

---

## Next steps (deliberately not built)

Known gaps, in the order they would matter:

- **Reranking.** A cross-encoder over the top ~50 candidates fixes the failure
  mode above: bi-encoder retrieval is cheap and approximate, a cross-encoder
  reads query and passage together and is far better at ranking near-duplicates.
- **Hybrid search.** BM25 (Postgres full-text) fused with vector results via
  reciprocal rank fusion. Exact strings — a ticker, a defined term, a statute —
  are precisely what dense retrieval is worst at.
- **HNSW instead of IVFFlat.** Better recall/latency at the cost of a slower
  build and more memory; IVFFlat is the right default at this corpus size.
- **Query expansion / decomposition** for multi-part questions.
- **Section-aware chunking** using 10-K item headings, so a chunk knows it is
  inside Item 1A rather than the MD&A.
- **Caching** of query embeddings, and streaming responses.
- Not built on purpose: a frontend, auth, rate limiting, cloud deployment.

## Licence

Code: MIT. The filings under `data/filings/` are public records retrieved from
SEC EDGAR and remain the works of their respective filers.
