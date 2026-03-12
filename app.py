import streamlit as st
import base64
from answer_policies_llama import answer_question
from query_policies import retrieve_policies
from pathlib import Path

tab_chat, tab_dates = st.tabs(["Chat", "Key Dates"])

PDF_DIR = Path("guidance pdf")

with tab_chat:
    st.set_page_config(page_title="Strathclyde Policy Assistant", layout="wide")

    st.title("Strathclyde Policy Assistant")
    st.caption("RAG demo: FAISS + MiniLM retrieval + Llama (Ollama) generation")

    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    
    with st.sidebar:
        st.header("Settings")
        k = 10
        st.markdown("---")
        st.markdown("**Tip:** Ask about exams, personal circumstances, admissions, marking.")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a policy question...")

    if prompt:
        # show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        
        chunks = retrieve_policies(prompt, k=k)

        # generate answer 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
               answer = answer_question(prompt, k=k)

        st.markdown(answer)

        with st.expander("Sources used", expanded=True):
            if not chunks:
                st.write("No sources retrieved.")
            else:
                for i, c in enumerate(chunks, start=1):
                    st.markdown(f"### Source {i}")
                    st.write(f"**Title:** {c.get('doc_title')}")
                    st.write(f"**Section:** {c.get('section')}")
                    if c.get("source_file"):
                        st.write(f"**PDF:** {c.get('source_file')}")
                        st.write(c.get("text", "")[:1500])


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
