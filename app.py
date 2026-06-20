import os
import re
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from rank_bm25 import BM25Okapi
from extract import extract_and_chunk_all

# 1. Page Configuration
st.set_page_config(page_title="UniRAG Academic Assistant", page_icon="📚", layout="wide")

# Helper function to tokenize clean alpha-numeric terms for BM25 matching
def tokenize_text(text):
    return re.findall(r'\b\w+\b', text.lower())

# Cache data loading & embedding generation so it only happens ONCE when the app starts
@st.cache_resource
def initialize_rag_engine():
    chunks_dataset = extract_and_chunk_all()
    if not chunks_dataset:
        return None, None, None, None
        
    texts = [item["text"] for item in chunks_dataset]
    
    # Setup Dense Index (FAISS)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Setup Sparse Index (BM25) with proper clean tokenization
    tokenized_corpus = [tokenize_text(text) for text in texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return chunks_dataset, model, index, bm25_index

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Run Hybrid Indexing
chunks_dataset, embedding_model, faiss_index, bm25_index = initialize_rag_engine()
reranker = load_reranker()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar Setup
# 2. Sidebar Setup
st.sidebar.title("📚 Course Navigation")

# Dynamic Drag & Drop File Uploader for Web Users
uploaded_files = st.sidebar.file_uploader(
    "Upload Course PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

# If a user uploads files through the browser, save them to the workspace
if uploaded_files:
    web_uploads_dir = os.path.join("data", "WEB_UPLOADS")
    os.makedirs(web_uploads_dir, exist_ok=True)
    
    new_file_added = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(web_uploads_dir, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_file_added = True
            
    # If new files were added, clear out the cached index and reload the RAG engine
    if new_file_added:
        st.cache_resource.clear()
        st.rerun()

# Build the dropdown course navigation checklist
if chunks_dataset:
    available_courses = sorted(list(set([item["metadata"]["course"] for item in chunks_dataset])))
    selected_course = st.sidebar.selectbox("Select Course Context:", available_courses)
else:
    st.sidebar.warning("No courses loaded yet. Drop a PDF above to get started!")
    selected_course = None

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.title("🎯 Smart Course Repository Chatbot")
st.caption("Ask questions grounded directly in your verified university lecture slides and notes.")

# Quick Check for API Key
if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing Gemini API Key. Please export your key in the terminal before running.")
else:
    client = genai.Client()

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat Input Interface
    if query := st.chat_input("Ask your course question here..."):
        
        with st.chat_message("user"):
            st.markdown(query)
        
        st.session_state.messages.append({"role": "user", "content": query})

        # Build conversational history context string
        history_context = ""
        if len(st.session_state.messages) > 1:
            history_context = "\nRecent Chat History Turn(s):\n"
            for msg in st.session_state.messages[-5:-1]:
                history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

        with st.spinner("Executing hybrid semantic and keyword lookup..."):
            tokenized_query = tokenize_text(query)
            
            # Fetch Top 15 from Dense Index (FAISS)
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            _, dense_indices = faiss_index.search(query_embedding, k=15)
            
            # Fetch Top 15 from Sparse Index (BM25)
            sparse_scores = bm25_index.get_scores(tokenized_query)
            sparse_indices = np.argsort(sparse_scores)[::-1][:15]
            
            # Combine unique candidates matching course code
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

            # Apply Cross-Encoder Re-ranking
            if candidate_chunks:
                pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
                scores = reranker.predict(pairs)
                ranked_indices = np.argsort(scores)[::-1]
                top_chunks = [candidate_chunks[i] for i in ranked_indices[:5]]
            else:
                top_chunks = []

            # Format the context string
            retrieved_context = ""
            citations = []
            for rank_counter, chunk in enumerate(top_chunks, 1):
                meta = chunk["metadata"]
                retrieved_context += f"\n[Document Context {rank_counter}]\nFile: {meta['filename']} | Page: {meta['page']}\nContent: {chunk['text']}\n"
                citations.append(f"Page {meta['page']} of {meta['filename']}")

            # 4. Synthesize with Gemini
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