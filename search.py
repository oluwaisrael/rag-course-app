import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from extract import extract_and_chunk_all

def main():
    # 1. Load data and embed (Same as before)
    print("Extracting and chunking PDF documents...")
    chunks_dataset = extract_and_chunk_all()
    if not chunks_dataset:
        print("No chunks found.")
        return
        
    texts = [item["text"] for item in chunks_dataset]
    
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 2. Initialize the new Gemini Client
    # It automatically looks for the GEMINI_API_KEY environment variable
    client = genai.Client()
    
    print("\n" + "="*50)
    print("FULL RAG GENERATION SYSTEM READY")
    print("Type 'exit' to quit.")
    print("="*50)
    
    while True:
        query = input("\nAsk your course question: ")
        if query.lower() == 'exit':
            break
            
        # 3. Retrieve the top 3 chunks from FAISS
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k=3)
        
        retrieved_context = ""
        citations = []
        
        for rank, idx in enumerate(indices[0]):
            match_idx = int(idx)
            if match_idx == -1:
                continue
            matched_chunk = chunks_dataset[match_idx]
            meta = matched_chunk["metadata"]
            
            # Format context block for the LLM
            retrieved_context += f"\n[Document Context {rank+1}]\n"
            retrieved_context += f"Course: {meta['course']} | File: {meta['filename']} | Page: {meta['page']}\n"
            retrieved_context += f"Content: {matched_chunk['text']}\n"
            
            # Save for displaying clean citations to the user
            citations.append(f"Page {meta['page']} of {meta['filename']} ({meta['course']})")
            
        # 4. Construct the RAG System Prompt
        system_instruction = (
            "You are an expert academic assistant for university undergraduate courses. "
            "Your task is to answer the user's question using ONLY the provided Document Context below. "
            "If the context does not contain the answer, say exactly 'I cannot find the answer in the provided course materials.' "
            "Do not make up facts or use outside knowledge."
        )
        
        user_prompt = f"""
Context materials:
{retrieved_context}

User Question: {query}

Answer the question thoroughly based on the context above.
"""

        # 5. Generate Answer using gemini-2.5-flash
        print("\nThinking...")
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config={'system_instruction': system_instruction}
            )
            
            print("\n=== ANSWER ===")
            print(response.text)
            print("\n=== SOURCES & CITATIONS ===")
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")
            print("="*30)
            
        except Exception as e:
            print(f"\nAPI Error: {e}")
            print("Make sure you ran: export GEMINI_API_KEY='your_key'")

if __name__ == "__main__":
    main()