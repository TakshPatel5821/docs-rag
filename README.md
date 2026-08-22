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
docker compose up -d --build      # first build downloads torch and bakes in the model
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

Ask the deployed service:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What does Apple say about its reliance on single or limited source suppliers?","top_k":3}'
```

The same question, run through the retrieval + grounding path without a database
(`eval/run_eval.py --ask`, exact cosine over the same chunks, `LLM_PROVIDER=echo`
so no credentials are involved). Real output, 2026-08-21:

```
4814 chunks | embedded in 151.7s | embedder=sentence-transformers

Q: What does Apple say about its reliance on single or limited source suppliers?

[echo provider -- no LLM configured; set LLM_PROVIDER to anthropic, openai or ollama
 for a generated answer]

Retrieved context that would have been used:
[1] source: AAPL_10-K_2025-10-31.txt (chunk 324, cosine similarity 0.640)
sufficient quantities from its suppliers or in a timely manner, or in identifying and
obtaining sufficient quantities from an alternative source.

In addition, component suppliers may fail, be subject to consolidation within a
particular industry, or decide to concentrate on the production of common components
instead of components customized to meet the Company's requirements, further limiting
the Company's ability to obtain sufficient quantities of components on commercially
reasonable terms, or at all.

Substantially all of the Company's hardware products are manufactured by outsourcing
partners that are located primarily in China mainland, India, Japan, South Korea,
Taiwan and Vietnam.

Apple Inc. | 2025 Form 10-K | 46

[2] source: AAPL_10-K_2025-10-31.txt (chunk 87, cosine similarity 0.630)
... Therefore, the Company remains subject to significant risks of supply shortages and
price increases that can materially adversely affect its business, results of
operations, financial condition and stock price.

Apple Inc. | 2025 Form 10-K | 8

[3] source: AAPL_10-K_2025-10-31.txt (chunk 98, cosine similarity 0.593)
The Company's future performance depends in part on support from third-party software
developers. ...

Sources:
  [1] AAPL_10-K_2025-10-31.txt (cosine 0.640)
  [2] AAPL_10-K_2025-10-31.txt (cosine 0.630)
  [3] AAPL_10-K_2025-10-31.txt (cosine 0.593)
```

Three things worth noticing, because they are what the design is actually about:

* All three passages come from the right filing out of four, and each answer
  carries the filing and chunk it came from. That is the attribution requirement.
* Passages `[1]` and `[2]` are **near-duplicates** — 10-Ks repeat the same risk
  language in Item 1A and again in the MD&A. One of the five slots is spent on a
  passage that adds nothing. This is the concrete argument for reranking and
  deduplication, further down.
* Set `LLM_PROVIDER=anthropic` (or `openai`/`ollama`) and the same passages go to a
  model under the grounding instruction; only the last hop changes.

### When the answer is not in the corpus

```
Q: Who won the 1998 FIFA World Cup final?

[1] BAC_10-K_2026-02-25.txt (chunk 726, cosine similarity 0.231)
[2] JPM_10-K_2026-02-13.txt (chunk 975, cosine similarity 0.175)
[3] JPM_10-K_2026-02-13.txt (chunk 1764, cosine similarity 0.169)
```

Vector search **always returns its k nearest neighbours** — there is no such thing
as "no result" — so an out-of-corpus question comes back with country-exposure
tables at cosine 0.17–0.23, against ~0.6 for a genuine hit. Two mechanisms handle
that:

1. the system instruction tells the model to answer only from the passages and to
   reply *"The provided context does not contain the answer to that question."*
   otherwise; and
2. when retrieval genuinely returns nothing (empty corpus, blank query), the
   service refuses **without calling the model at all**.

The gap between 0.23 and 0.6 is also the argument for a calibrated similarity
floor — listed under next steps rather than shipped with an invented threshold.

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

Measured on the committed corpus: **4,814 chunks embedded in 137 s** on a laptop
CPU (Python 3.13, torch 2.13 CPU-only), about 35 chunks/second. Ingestion is a
one-off; a query embeds a single string in milliseconds.

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

Ten questions with known answers live in `eval/questions.json`, each tagged with
the filing that actually contains the answer and a term the right passage must
contain. `eval/run_eval.py` scores two things at k=5:

* **source recall@5** — did any of the five retrieved chunks come from the right
  filing? If the right document never reaches the context window, the generator
  cannot be right except by luck.
* **keyword hit@5** — did a retrieved chunk from the right filing also contain the
  expected term, i.e. did we get the right *passage* and not just the right
  *document*?

```bash
python eval/run_eval.py                     # in-memory, no database needed
python eval/run_eval.py --backend pgvector  # against the live stack, incl. IVFFlat
```

At the shipped defaults (800/150, MiniLM, k=5), 4,814 chunks:

```
source recall@5: 100%   keyword hit@5: 80%   MRR: 0.933
```

The two questions that retrieve the right filing but miss the expected term
("how much did Apple spend on R&D", "Microsoft capital expenditures") both ask for
a **number that lives in a table**. Financial tables flatten into runs of digits
with the row labels far away, and a bi-encoder embedding of "4,516 1,320 607" says
almost nothing. Table-aware extraction, not a bigger model, is the fix.

### Chunk size: the sweep

`python eval/run_eval.py --sweep` re-chunks and re-embeds the whole corpus at each
setting and re-scores the same ten questions:

| chunk | overlap | chunks | recall@5 | keyword@5 | MRR |
|------:|--------:|-------:|---------:|----------:|----:|
| 400 | 75 | 10,073 | 100% | 80% | 0.875 |
| **800** | **150** | **4,814** | **100%** | **80%** | **0.933** |
| 1200 | 200 | 3,097 | 100% | 80% | 1.000 |
| 2000 | 200 | 1,661 | 100% | 80% | 1.000 |

Read honestly: **the metric saturates**. Every configuration finds the right filing
every time, so recall cannot separate them, and ten questions is far too small a
sample to treat an MRR difference of 0.07 as a real effect. What the sweep does
establish is a floor — 400/75 ranks the right passage lower, which is the expected
"too small loses context" failure — and that the choice is between 800 and
something larger.

800/150 ships because larger chunks buy their MRR with context window: at 2000
characters, five passages is ~10,000 characters of prompt per query, most of it
irrelevant, and every retrieved chunk that is 60% padding is padding the model has
to read past. The honest interview answer is "800 is the smallest setting that
loses nothing measurable on my eval set, and my eval set is ten questions" — not
"800 is optimal".

The obvious next move is a bigger, harder question set, including questions whose
answers straddle chunk boundaries, where overlap actually earns its keep.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

70 tests, no services required.

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

- **Reranking.** A cross-encoder over the top ~50 candidates, keeping 5. This
  addresses the duplicate-passage waste seen in the worked example above: a
  bi-encoder scores query and passage separately and cannot tell that `[2]` adds
  nothing over `[1]`, while a cross-encoder reads them together.
- **Table-aware extraction.** Both keyword misses in the evaluation are numbers
  living in financial tables, which flatten into meaningless digit runs. Parsing
  tables into labelled rows before chunking would fix a real, measured failure —
  the highest-value item on this list.
- **A calibrated similarity floor.** Out-of-corpus questions return neighbours at
  cosine ~0.2 against ~0.6 for genuine hits, so a threshold could refuse before
  spending a model call. It needs calibrating against a labelled set rather than
  a guessed constant, which is why it is here and not in the code.
- **Hybrid search.** BM25 (Postgres full-text) fused with vector results by
  reciprocal rank fusion. Exact strings — a ticker, a defined term, a statute —
  are precisely what dense retrieval is worst at.
- **HNSW instead of IVFFlat.** Better recall/latency, at the cost of slower builds
  and more memory; IVFFlat is the right default at this corpus size.
- **Section-aware chunking** using 10-K item headings, so a chunk knows it sits in
  Item 1A rather than the MD&A, and **query decomposition** for multi-part questions.
- **A bigger evaluation set.** Ten questions cannot separate the chunking configs;
  the sweep saturates. More questions, and harder ones.
- **Caching** of query embeddings, and streaming responses.
- Not built on purpose: a frontend, auth, rate limiting, cloud deployment.

## Licence

Code: MIT. The filings under `data/filings/` are public records retrieved from
SEC EDGAR and remain the works of their respective filers.
