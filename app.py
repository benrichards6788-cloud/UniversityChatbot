import base64
from collections import defaultdict
from pathlib import Path

import streamlit as st

from answer_policies_llama import answer_question
from query_policies import retrieve_policies

PDF_DIR = Path("guidance pdf")
FIXED_K = 10
MAX_EXTRACT_CHARS = 700


def render_pdf_embed(pdf_path: Path, height: int = 700):
    """Embed a local PDF inside the Streamlit app."""
    if not pdf_path.exists():
        st.warning(f"PDF not found: {pdf_path.name}")
        return

    pdf_bytes = pdf_path.read_bytes()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="{height}"
            type="application/pdf"
            style="border: 1px solid #444; border-radius: 8px;">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def strip_breadcrumbs(text: str) -> str:
    """Remove leading [Policy: ...] [Section: ...] breadcrumbs from chunk text for UI display."""
    if not text:
        return ""

    cleaned = text.strip()
    if cleaned.startswith("[Policy:"):
        parts = cleaned.split("]", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return cleaned


def group_chunks_by_document(chunks):
    grouped = defaultdict(list)
    for chunk in chunks or []:
        doc_title = chunk.get("doc_title") or "Unknown document"
        grouped[doc_title].append(chunk)
    return grouped


st.set_page_config(page_title="Strathclyde Policy Assistant", layout="wide")

tab_chat, tab_dates = st.tabs(["Chat", "Key Dates"])

with tab_chat:
    st.title("Strathclyde Policy Assistant")
    st.caption("RAG demo: FAISS + MiniLM retrieval + Llama (Ollama) generation")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        st.markdown("---")
        st.markdown("**Tip:** Ask about exams, personal circumstances, admissions, marking.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a policy question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        chunks = retrieve_policies(prompt, k=FIXED_K)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(prompt, k=FIXED_K)

            st.markdown(answer)

            with st.expander("Sources used", expanded=True):
                if not chunks:
                    st.write("No sources retrieved.")
                else:
                    grouped_chunks = group_chunks_by_document(chunks)

                    for source_num, (doc_title, doc_chunks) in enumerate(grouped_chunks.items(), start=1):
                        first_chunk = doc_chunks[0]
                        source_file = first_chunk.get("source_file")
                        section_names = sorted({(c.get("section") or "Unknown section") for c in doc_chunks})

                        st.markdown(f"### 📄 Source {source_num}")
                        st.write(f"**Document:** {doc_title}")
                        st.write(f"**Retrieved excerpts:** {len(doc_chunks)}")

                        if section_names:
                            if len(section_names) == 1:
                                st.write(f"**Section:** {section_names[0]}")
                            else:
                                st.write(f"**Sections:** {', '.join(section_names[:3])}")

                        if source_file:
                            st.write(f"**PDF:** {source_file}")

                        for extract_num, chunk in enumerate(doc_chunks, start=1):
                            chunk_text = strip_breadcrumbs(chunk.get("text", ""))
                            st.markdown(f"**Relevant extract {extract_num}:**")
                            st.info(chunk_text[:MAX_EXTRACT_CHARS])

                        if source_file:
                            pdf_path = PDF_DIR / source_file
                            show_pdf = st.toggle(
                                f"View source PDF: {source_file}",
                                key=f"toggle_pdf_{source_num}_{source_file}"
                            )
                            if show_pdf:
                                render_pdf_embed(pdf_path)

                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_dates:
    st.subheader("University Key Dates")
    st.write(
        "These are the official academic key dates published by the University. "
        "For the most up-to-date information, always consult the official website."
    )

    st.link_button(
        "Open Strathclyde Key Dates",
        "https://www.strath.ac.uk/keydates/"
    )

    st.markdown("### Academic Year Overview (example)")
    st.markdown("- Welcome and Development Week: September")
    st.markdown("- Semester 1 teaching block")
    st.markdown("- Winter assessment period")
    st.markdown("- Semester 2 teaching block")
    st.markdown("- Spring assessment period")
    st.caption(
        "Exact dates vary by academic year and programme. "
        "This prototype surfaces official information rather than storing it locally."
    )
