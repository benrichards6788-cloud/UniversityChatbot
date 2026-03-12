import base64
from collections import defaultdict
from pathlib import Path
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from answer_policies_llama import answer_question
from query_policies import retrieve_policies

PDF_DIR = Path("guidance pdf")
FIXED_K = 10
MAX_EXTRACT_CHARS = 500


def render_pdf_embed(pdf_path: Path, height: int = 700):
    if not pdf_path.exists():
        st.warning(f"PDF not found: {pdf_path}")
        return

    pdf_bytes = pdf_path.read_bytes()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_html = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="{height}"
        style="border: none; border-radius: 8px;">
    </iframe>
    """
    components.html(pdf_html, height=height + 20, scrolling=True)


def strip_breadcrumbs(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    if cleaned.startswith("[Policy:"):
        parts = cleaned.split("]", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return cleaned


def clean_section_name(section: str) -> str:
    if not section:
        return ""

    section = section.strip()

    if section.lower() == "document":
        return ""

    tokens = section.split()
    if len(tokens) >= 6 and sum(len(tok) <= 2 for tok in tokens) / len(tokens) >= 0.6:
        return ""

    return " ".join(tokens)


def group_chunks_by_document(chunks):
    grouped = defaultdict(list)
    for chunk in chunks or []:
        doc_title = chunk.get("doc_title") or "Unknown document"
        grouped[doc_title].append(chunk)
    return grouped


st.set_page_config(page_title="Strathclyde Policy Assistant", layout="wide")

st.markdown(
        """
        <style>
            .main .block-container {
                padding-bottom: 8rem;
            }

            div[data-testid="stChatInput"] {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                width: 100%;
                z-index: 1000;
                padding: 1rem 2rem;
                background: transparent;
            }

            div[data-testid="stChatInput"] > div {
                width: 100%;
                max-width: 1200px;
                margin: auto;
                background: rgb(14,17,23);
                border-radius: 14px;
                box-shadow: 0 6px 24px rgba(0,0,0,0.35);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
""",
unsafe_allow_html=True,
)

tab_chat, tab_dates = st.tabs(["Chat", "Key Dates"])

with tab_chat:
    st.title("Strathclyde Policy Assistant")
    st.caption("Ask me anything about Strathclyde policy and I will assist you!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources_expanded" not in st.session_state:
        st.session_state.sources_expanded = False

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
        st.session_state.sources_expanded = False

        with st.chat_message("user"):
            st.markdown(prompt)

        chunks = retrieve_policies(prompt, k=FIXED_K)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(prompt, k=FIXED_K)

            st.markdown(answer)

            with st.expander("Sources used", expanded=st.session_state.sources_expanded):
                if not chunks:
                    st.write("No sources retrieved.")
                else:
                    grouped_chunks = group_chunks_by_document(chunks)

                    for source_num, (doc_title, doc_chunks) in enumerate(grouped_chunks.items(), start=1):
                        first_chunk = doc_chunks[0]
                        source_file = first_chunk.get("source_file")

                        raw_sections = [c.get("section") or "" for c in doc_chunks]
                        section_names = sorted(
                            {
                                clean_section_name(section)
                                for section in raw_sections
                                if clean_section_name(section)
                            }
                        )

                        st.markdown(f"### 📄 {doc_title}")
                        st.caption(
                            f"{len(doc_chunks)} supporting excerpt"
                            + ("s" if len(doc_chunks) > 1 else "")
                        )

                        if section_names:
                            if len(section_names) == 1:
                                st.write(f"**Section:** {section_names[0]}")
                            else:
                                st.write(f"**Sections:** {', '.join(section_names[:3])}")

                        if source_file:
                            st.write(f"**PDF:** {source_file}")

                        for extract_num, chunk in enumerate(doc_chunks, start=1):
                            chunk_text = strip_breadcrumbs(chunk.get("text", ""))
                            safe_chunk = escape(chunk_text[:MAX_EXTRACT_CHARS])

                            st.markdown(f"**Retrieved Evidence {extract_num}:**")
                            st.markdown(
                                f"""
                                <div style="
                                background-color:#1e3a5f;
                                padding:14px;
                                border-radius:8px;
                                border-left:4px solid #4da3ff;
                                font-size:0.95rem;
                                line-height:1.5;">
                                {safe_chunk}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if source_file:
                            pdf_path = PDF_DIR / source_file
                            pdf_key = f"show_pdf_{source_num}_{source_file}"

                            if pdf_key not in st.session_state:
                                st.session_state[pdf_key] = False

                            col1, col2 = st.columns([1, 1])

                            with col1:
                                if pdf_path.exists():
                                    with open(pdf_path, "rb") as f:
                                        st.download_button(
                                            "Download PDF",
                                            data=f.read(),
                                            file_name=source_file,
                                            mime="application/pdf",
                                            key=f"dl_{pdf_key}"
                                        )

                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

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
