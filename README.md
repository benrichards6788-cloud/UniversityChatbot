````md
# Strathclyde Policy Assistant — README

**Author:** Ben Richards  
**Course:** CS408 Individual Project, University of Strathclyde

> **A live deployed version of this project is available and can be used directly without any setup:**  
> https://universitychatbot-strathclyde.streamlit.app

---

## Requirements

- Python 3.10+
- An OpenAI API key (required for answer generation)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
````

Create a `.env` file in the project root containing your API key:

```
OPENAI_API_KEY=your_key_here
```

---

## Running the app (Recommended)

```bash
streamlit run app.py
```

The chatbot will open in your browser. Type a policy question and press Enter.

The system is ready to use immediately — a pre-built FAISS index and processed data are included in this submission.

---

## Building the index (Optional)

The index does not need to be rebuilt to run the system.

If you wish to recreate the pipeline from raw PDFs:

Place policy PDFs in a folder called `guidance pdf/`, then run:

```bash
python load_pdf.py
python sem_chunker.py
python index_policies.py
```

This will:

* Extract text from PDFs
* Generate semantic chunks
* Build the FAISS vector index

---

## Included data

This submission already contains all required data files:

* `chunks_semantic.jsonl` (processed policy chunks)
* `vectorstore/policy.index` (FAISS index)
* `vectorstore/meta.json` (metadata)

---

## Running the tests

```bash
pytest test_pipeline.py -v
```

89 tests covering unit, integration, and system levels.

---

## Notes

* The system uses hybrid retrieval (FAISS + BM25)
* Answers are generated using a large language model grounded in retrieved policy text
* Source excerpts are displayed to improve transparency and trust

```
