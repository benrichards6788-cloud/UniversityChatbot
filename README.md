# Strathclyde Policy Assistant — README

**Author:** Ben Richards  
**Course:** CS408 Individual Project, University of Strathclyde

> **A live deployed version of this project is available and can be used directly without any setup:**  
> https://universitychatbot-strathclyde.streamlit.app

---

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root containing your API key:

```
OPENAI_API_KEY=your_key_here
```

## Building the index

Place policy PDFs in a folder called `guidance pdf/`, then run these three scripts in order:

```bash
python load_pdf.py
python sem_chunker.py
python index_policies.py
```

This extracts text from the PDFs, chunks them semantically, and builds the FAISS vector index.

## Running the app

```bash
streamlit run app.py
```

The chatbot will open in your browser. Type a policy question and press Enter.

## Running the tests

```bash
pytest test_pipeline.py -v
```

89 tests covering unit, integration, and system levels.
