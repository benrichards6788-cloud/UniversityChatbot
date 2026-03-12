"""
semantic chunker python file
splits chunks of text based on their meaning
produces a JSONL file 

"""

import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Iterable

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# configuration defaults

DEFAULT_INPUT_DIR = "rawpdf"                 # raw input
DEFAULT_OUTPUT_JSONL = "data/chunks_semantic.jsonl"
DEFAULT_TARGET_TOKENS = 900                  # specific design choice outlined in notes
DEFAULT_OVERLAP_RATIO = 0.12                 # parameter placeholder, current implementation uses one sentence overlap
DEFAULT_SIM_THRESHOLD = 0.60                 # sentence-to-sentence similarity split
DEFAULT_MAX_RUN = 8                          # max consecutive sentences per semantic run

_HEADING_MAX_LEN = 120

def _normalize_heading(h: str) -> str:
    h = h.replace("\n", " ")
    h = re.sub(r"\s+", " ", h).strip()
    # trim trailing punctuation that isn't part of title
    h = h.rstrip(":;.,")
    return h

def _looks_plausible_heading(h: str) -> bool:
    """
    reject headings that are likely body text
    """
    if not h:
        return False
    if len(h) > _HEADING_MAX_LEN:
        return False

    # if ends with a period and is long, likely a sentence not a heading
    if h.endswith(".") and len(h.split()) > 12:
        return False

    # ratio of lowercase letters
    letters = re.findall(r"[A-Za-z]", h)
    if letters:
        lower_ratio = sum(ch.islower() for ch in letters) / len(letters)
        if lower_ratio > 0.4 and not h.endswith(":"):
            return False

    # numbered line that immediately continues with lowercase
    if re.match(r"^\d+(?:\.\d+){0,3}\s+[a-z]", h):
        return False

    return True





# token estimation (fast)

def approx_tokens(text: str) -> int:
    """
    turns the text to an estimated no. of tokens
    english text avg 1 token = 4 chars (simple approximation)
    """
    return max(1, math.ceil(len(text) / 4))

# light cleaning

def clean_text(text: str) -> str:
    """
    cleans up common pdf formatting issues
    """
    # remove Windows carriage return (invis characters)
    text = text.replace("\r", "")

    # remove our own file merge markers (if any)
    text = re.sub(r"(?m)^\s*===== FILE: .+? =====\s*$", "", text)

    # remove earlier page separators if present
    text = re.sub(r"\n?--- PAGE \d+ ---\n?", "\n", text)

    # de-hyphenate line breaks (word-<newline>word -> wordword)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # replace bullets like • with hyphen for consistency
    text = re.sub(r"\n[•▪•·]\s*", "\n- ", text)

    # flatten single newlines inside paragraphs: X\nY -> X Y
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)

    # normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # remove dotted leader page refs like: "Title ..... 12"
    text = re.sub(r'\s?\.{2,}\s?\d+\b', '', text)

    # remove lines that are just (page) numbers 
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # remove a few known recurring boilerplates
    text = re.sub(r"(?mi)^the place of useful learning.*$", "", text)
    text = re.sub(r"(?mi)^.*charitable body.*SC0?15263.*$", "", text)

    # collapse 3+ newlines to 2 (paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# section detection (heuristic)

"""multi line regular expression that looks for lines that resemble section headings"""
_SECTION_RE = re.compile(
    r"(?m)^(?P<h>"                  # start line, capture group 'h'
    r"[A-Z][A-Z \-/&]{3,}"          # matches lines written entirely in uppercase
    r"|"                            # — (or)
    r"(?:\d+(?:\.\d+){0,3})\s+[^\n:]{3,}"  # numbered headings: 1 / 1.1 / 2.4.3 etc
    r"|"                            # — (or)
    r"[A-Z].{2,50}:"                # short title: (ends with colon)
    r")\s*$"                        # allow spaces right after the heading
)

def split_sections(text: str) -> List[Dict]:
    """
    matches each heading with the text that belongs to that heading
    it returns a list of {title, body}
    if no headings are detected -> return a single 'Document' section
    """
    parts = []
    last = 0
    current_title = "Document"

    for m in _SECTION_RE.finditer(text):
        start = m.start()

        if start > last:
            body = text[last:start].strip()
            if body:
                parts.append({"title": current_title, "body": body})

        candidate = _normalize_heading(m.group("h"))
        if _looks_plausible_heading(candidate):
            current_title = candidate
            last = m.end()
        else:
        # treat rejected heading as body text
            last = m.start()


    # tail body after the final heading
    tail = text[last:].strip()
    if tail:
        parts.append({"title": current_title, "body": tail})

    # drop obvious non-content sections like Contents/Version
    filtered = []
    for sec in parts:
        t = sec["title"]
        if re.search(r"(?i)\bcontents\b", t):
            continue
        if re.search(r"(?i)\bversion\s*no\.?\b", t):
            continue
        filtered.append(sec)

    return filtered if filtered else [{"title": "Document", "body": text}]


def merge_small_sections(
    sections: List[Dict],
    min_tokens: int = 120,
    join_with_next: bool = True,
) -> List[Dict]:
    """
    merge small sections with their neighbors.
    """
    if not sections:
        return sections

    # choose direction; default is forward-merge 
    out: List[Dict] = []
    if join_with_next:
        buf = None
        for i, s in enumerate(sections):
            body = (s.get("body") or "").strip()
            if not body:
                continue

            if buf is None:
                buf = dict(s)
                continue

            if approx_tokens(buf["body"]) < min_tokens:
                # merge small previous into current
                merged = {
                    "title": s["title"],
                    "body": (buf["body"].strip() + "\n\n" + body).strip(),
                }
                buf = merged
            else:
                out.append(buf)
                buf = dict(s)
        if buf:
            out.append(buf)
        return out

    else:
        # backward-merge; add small section into previous
        for s in sections:
            body = (s.get("body") or "").strip()
            if not body:
                continue
            if out and approx_tokens(body) < min_tokens:
                out[-1]["body"] = (out[-1]["body"].strip() + "\n\n" + body).strip()
            else:
                out.append({"title": s["title"], "body": body})
        return out


_SUBSECTION_INLINE_RE = re.compile(
    r"(?=(\b\d+\.)\s+([A-Z][A-Z \-/&]{3,})(?=\s+\d+\.\d+))"
)

def split_subsections(body: str, parent_title: str, min_tokens: int = 500) -> List[Dict]:
    """split a section body into smaller subsections based on inline all-caps numbered headings.

    returns a list of {title, body}. If no split points are found, returns one item.
    """
    body = (body or "").strip()
    if not body:
        return []

    # only bother when the body is large enough that dilution is likely
    if approx_tokens(body) < min_tokens:
        return [{"title": parent_title, "body": body}]

    matches = list(_SUBSECTION_INLINE_RE.finditer(body))
    if not matches:
        return [{"title": parent_title, "body": body}]

    # build segments between match starts
    starts = [m.start() for m in matches]
    starts.append(len(body))

    out: List[Dict] = []
    # optional prefix before the first detected subheading
    if starts[0] > 0:
        prefix = body[:starts[0]].strip()
        if prefix:
            out.append({"title": parent_title, "body": prefix})

    for i, m in enumerate(matches):
        seg_start = m.start()
        seg_end = starts[i + 1]
        seg = body[seg_start:seg_end].strip()
        if not seg:
            continue
        num = m.group(1).strip()
        name = _normalize_heading(m.group(2))
        sub_title = f"{parent_title} | {num} {name}"
        out.append({"title": sub_title, "body": seg})

    return out if out else [{"title": parent_title, "body": body}]




# sentence splitting (simple, robust)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])") #split when ".", !, or ? followed by space and then a capital, number, or "("

def split_sentences(text: str) -> List[str]:
    """
    simple but effective for policy text
    also keeps bullet points together 
    """
    sents = _SENT_END.split(text)
    return [s.strip() for s in sents if s and s.strip()]


# semantic grouping with MiniLM


# load MiniLM once (small, fast on CPU)
_EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_sentences(sentences: List[str]):
    # convert to tensor and normalize for efficient cosine similarity
    embs = _EMB.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    return embs

def group_semantically(
    sentences: List[str],
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    max_run: int = DEFAULT_MAX_RUN,
) -> List[List[str]]:
    """
    builds semantic runs by scanning consecutive sentences and
    starting a new run when cohesion drops below 0.6 similarity or
    after 8 runs.
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[:]]

    embs = encode_sentences(sentences)
    runs = []
    start = 0
    for i in range(1, len(sentences)):
        sim = float(util.cos_sim(embs[i-1], embs[i]))
        # new run on topic shift (low sim) or to cap very long runs
        if sim < sim_threshold or (i - start) >= max_run:
            runs.append(sentences[start:i])
            start = i
    runs.append(sentences[start:])
    return runs


# pack runs into chunks with overlap


def pack_runs_to_chunks(
    runs: List[List[str]],
    doc_title: str,
    section_title: str,
    source_file: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    min_tokens: int = 120
) -> List[Dict]:
    """
    packs semantic runs into chunks ~target_tokens.
    adds breadcrumb headers for semantic independence.
    adds ~1 sentence overlap between adjacent chunks.
    """
    chunks = []
    cur_buf: List[str] = [] #current chunk being filled
    cur_tok = 0 #how full the current chunk is (est. tokens)

    for run in runs:
        run_text = " ".join(run).strip()
        rtok = approx_tokens(run_text)

        # if adding this run would exceed budget + chunk not empty, finish chunk
        if cur_buf and (cur_tok + rtok > target_tokens):
            if cur_buf:
                chunks.append(_make_chunk(doc_title, section_title, source_file, cur_buf))
            cur_buf, cur_tok = [], 0

        cur_buf.append(run_text)
        cur_tok += rtok

    # after the loop, seal the last part into a final chunk
    if cur_buf:
        chunks.append(_make_chunk(doc_title, section_title, source_file, cur_buf))

    # duplicate the first ~1 sentence of the next chunk
    out = []
    for i, ch in enumerate(chunks):
        if i > 0:
            prev = out[-1]
            body = chunk_body(ch["text"])
            first_sent = split_sentences(body)[:1]
            if first_sent:
                prev["text"] = prev["text"].rstrip() + " " + first_sent[0]
        out.append(ch)

    # add token estimate (final)
    for ch in out:
        ch["tokens_est"] = approx_tokens(ch["text"])

    if len(out) >= 2:
        last = out[-1]
        if last["tokens_est"] < min_tokens : 
            second_last = out[-2]
            last_body = chunk_body(last["text"])
            second_last["text"] = second_last["text"].rstrip() + "\n\n" + last_body.strip()
            second_last["tokens_est"] = approx_tokens(second_last["text"])
            out.pop()

    return out

def _make_chunk(doc_title: str, section_title: str, source_file: str, pieces: Iterable[str]) -> Dict:
    breadcrumb = f"[Policy: {doc_title}] [Section: {section_title}]"
    text = breadcrumb + "\n\n" + " ".join(pieces).strip()
    return {
        "doc_title": doc_title,
        "section": section_title,
        "text": text,
        "source_file": source_file,
    }

def chunk_body(text_with_breadcrumb: str) -> str:
    """
    returns body text after the 2 newlines that follow the breadcrumb
    used so overlap copies body content (not header)
    """
    parts = text_with_breadcrumb.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text_with_breadcrumb


# chunk a single file


def chunk_file(
    txt_path: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    max_run: int = DEFAULT_MAX_RUN,
) -> List[Dict]:
    """
    loads 1 txt doc, removes page markers, cleans it,
    splits into sections -> sentences -> semantic runs, then packs into chunks.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")

    cleaned = clean_text(raw)

    sections = split_sections(cleaned)
    sections = merge_small_sections(
        sections,
        min_tokens=120,          
        join_with_next=True  
    )

    doc_title = prettify_title(txt_path.stem)
    source_file = txt_path.with_suffix(".pdf").name

    all_chunks: List[Dict] = []

    for sec in sections:
        body = sec["body"]
        if not body:
            continue

    # split large sections into numbered sub-sections (e.g. 4. ANONYMOUS MARKING)
        sub_sections = split_subsections(
            body,
            parent_title=sec["title"],
            min_tokens=500
        )

        for sub in sub_sections:
            sub_body = sub["body"]
            if not sub_body:
                continue

            sentences = split_sentences(sub_body)
            if not sentences:
                continue

            runs = group_semantically(
                sentences,
                sim_threshold=sim_threshold,
                max_run=max_run
            )

            packed = pack_runs_to_chunks(
                runs,
                doc_title=doc_title,
                section_title=sub["title"],   # important: use subsection title
                source_file=source_file,
                target_tokens=target_tokens,
                overlap_ratio=overlap_ratio,
            )

            all_chunks.extend(packed)


    return all_chunks

def prettify_title(stem: str) -> str:
    title = stem.replace("_", " ").replace("-", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title if title.isupper() else title.title()


# main: iterate folder, write JSONL

def chunk_folder(
    input_dir: Path,
    out_jsonl: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    max_run: int = DEFAULT_MAX_RUN,
):
    input_dir = Path(input_dir)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count_files = 0
    count_chunks = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in tqdm(sorted(input_dir.glob("*.txt")), desc="Chunking files"):
            chunks = chunk_file(
                p,
                target_tokens=target_tokens,
                overlap_ratio=overlap_ratio,
                sim_threshold=sim_threshold,
                max_run=max_run,
            )
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
                count_chunks += 1
            count_files += 1

    print(f"\nProcessed {count_files} files. Wrote {count_chunks} chunks to {out_jsonl}")


# CLI

def parse_args():
    ap = argparse.ArgumentParser(description="Semantic chunker for policy text files.")
    ap.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Folder with .txt files")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path")
    ap.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    ap.add_argument("--overlap_ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--max_run", type=int, default=DEFAULT_MAX_RUN)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    chunk_folder(
        input_dir=Path(args.input),
        out_jsonl=Path(args.output),
        target_tokens=args.target_tokens,
        overlap_ratio=args.overlap_ratio,
        sim_threshold=args.sim_threshold,
        max_run=args.max_run,
    )
