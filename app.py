import base64
from pathlib import Path

import streamlit as st

from answer_policies_llama import answer_question
from query_policies import retrieve_policies

PDF_DIR = Path("guidance pdf")


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


st.set_page_config(page_title="Strathclyde Policy Assistant", layout="wide")

tab_chat, tab_dates = st.tabs(["Chat", "Key Dates"])

with tab_chat:
    st.title("Strathclyde Policy Assistant")
    st.caption("RAG demo: FAISS + MiniLM retrieval + Llama (Ollama) generation")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        k = 10
        st.markdown("---")
        st.markdown("**Tip:** Ask about exams, personal circumstances, admissions, marking.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a policy question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        chunks = retrieve_policies(prompt, k=k)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(prompt, k=k)

            st.markdown(answer)

            with st.expander("Sources used", expanded=True):
                if not chunks:
                    st.write("No sources retrieved.")
                else:
                    shown_pdfs = set()

                    for i, c in enumerate(chunks, start=1):
                        st.markdown(f"### Source {i}")
                        st.write(f"**Title:** {c.get('doc_title')}")
                        st.write(f"**Section:** {c.get('section')}")

                        source_file = c.get("source_file")
                        if source_file:
                            st.write(f"**PDF:** {source_file}")

                        st.markdown("**Relevant extract:**")
                        st.info(c.get("text", "")[:1200])

                        if source_file:
                            pdf_path = PDF_DIR / source_file

                            # only offer one embed toggle per unique PDF
                            if source_file not in shown_pdfs:
                                shown_pdfs.add(source_file)

                                show_pdf = st.toggle(
                                    f"View source PDF: {source_file}",
                                    key=f"toggle_pdf_{source_file}"
                                )

                                if show_pdf:
                                    render_pdf_embed(pdf_path)

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