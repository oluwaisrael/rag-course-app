import os
import re
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from rank_bm25 import BM25Okapi
from extract import extract_and_chunk_all


st.set_page_config(page_title="Derin's Academic Assistant", page_icon="📚", layout="wide")

def tokenize_text(text):
    return re.findall(r'\b\w+\b', text.lower())


@st.cache_resource
def initialize_rag_engine():
    chunks_dataset = extract_and_chunk_all()
    if not chunks_dataset:
        return None, None, None, None
        
    texts = [item["text"] for item in chunks_dataset]
    

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    tokenized_corpus = [tokenize_text(text) for text in texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return chunks_dataset, model, index, bm25_index

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

chunks_dataset, embedding_model, faiss_index, bm25_index = initialize_rag_engine()
reranker = load_reranker()

if chunks_dataset and len(chunks_dataset) >= 1000:
    st.sidebar.warning("Haba nau, it's too much... derin didnt pay for this!")

if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.title(" These are Your Courses man!")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)


if uploaded_files:
    web_uploads_dir = os.path.join("data", "Not Derin's media!")
    os.makedirs(web_uploads_dir, exist_ok=True)
    
    new_file_added = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(web_uploads_dir, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_file_added = True

    if new_file_added:
        st.cache_resource.clear()
        st.rerun()


if chunks_dataset:
    available_courses = sorted(list(set([item["metadata"]["course"] for item in chunks_dataset])))
    selected_course = st.sidebar.selectbox("from here?:", available_courses)
else:
    st.sidebar.warning("No courses loaded yet. Drop a PDF above to get started!")
    selected_course = None

if st.sidebar.button("DUMP YOUR EX! in the binnnn!"):
    st.session_state.messages = []
    st.rerun()

st.title("🤷🏾‍♂️Derin made an ai assistant for you!(maybe for himself)🤷🏾‍♂️")
st.caption("DO NOT ASK OUTSIDE THE FILE YOU UPLOADED ABEG!😭")

# Quick Check for API Key
if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing Gemini API Key. Please export your key in the terminal before running.")
else:
    client = genai.Client()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

 
    if query := st.chat_input("Oya what's today's question, dumbasssss"):
        
        with st.chat_message("user"):
            st.markdown(query)
        
        st.session_state.messages.append({"role": "user", "content": query})

        history_context = ""
        if len(st.session_state.messages) > 1:
            history_context = "\nRecent Chat History Turn(s):\n"
            for msg in st.session_state.messages[-5:-1]:
                history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

        with st.spinner("Executing hybrid semantic and keyword lookup..."):
            tokenized_query = tokenize_text(query)
            

            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            _, dense_indices = faiss_index.search(query_embedding, k=15)
            

            sparse_scores = bm25_index.get_scores(tokenized_query)
            sparse_indices = np.argsort(sparse_scores)[::-1][:15]
            

            candidate_pool_indices = set(dense_indices[0]).union(set(sparse_indices))
            candidate_chunks = []
            
            for idx in candidate_pool_indices:
                match_idx = int(idx)
                if match_idx == -1 or match_idx >= len(chunks_dataset):
                    continue
                matched_chunk = chunks_dataset[match_idx]
                meta = matched_chunk["metadata"]
                
                if meta["course"] == selected_course:
                    candidate_chunks.append(matched_chunk)


            if candidate_chunks:
                pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
                scores = reranker.predict(pairs)
                ranked_indices = np.argsort(scores)[::-1]
                top_chunks = [candidate_chunks[i] for i in ranked_indices[:5]]
            else:
                top_chunks = []

            retrieved_context = ""
            citations = []
            for rank_counter, chunk in enumerate(top_chunks, 1):
                meta = chunk["metadata"]
                retrieved_context += f"\n[Document Context {rank_counter}]\nFile: {meta['filename']} | Page: {meta['page']}\nContent: {chunk['text']}\n"
                citations.append(f"Page {meta['page']} of {meta['filename']}")

            #gemini
            if retrieved_context:
                system_instruction = (
                    "You are an expert academic professor and teaching assistant. Your job is to answer the user's current question "
                    "by fully leveraging, summarizing, and explaining the facts found within the provided Document Context. "
                    "Be insightful, structure your response clearly with bullet points where necessary, and reference details accurately. "
                    "Use the provided Chat History Context to track follow-up conversational topics and pronouns smoothly."
                )
                
                user_prompt = f"Chat History Context:\n{history_context}\n\nDocument Context:\n{retrieved_context}\n\nCurrent Question: {query}"
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=user_prompt,
                        config={'system_instruction': system_instruction}
                    )
                    
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                        
                        if citations:
                            st.markdown("---")
                            st.markdown("**📚 Sources Verified (Hybrid Search + Re-ranked):**")
                            for citation in list(set(citations)):
                                st.caption(f"• {citation}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
                except Exception as e:
                    st.error(f"API Error: {e}")
            else:
                with st.chat_message("assistant"):
                    st.warning(f"No specific documentation found matching that query under course code {selected_course}.")