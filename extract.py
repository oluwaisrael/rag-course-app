import os
import pdfplumber

def chunk_text_recursively(text, max_chunk_size=1200, overlap=200):
    """
    Splits text dynamically by trying to keep paragraphs and sentences together.
    """

    separators = ["\n\n", "\n", " ", ""]
    chunks = []

    current_chunk = ""
    words = text.split(" ")
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chunk_size:
            current_chunk += (word + " ")
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Retain overlap from the end of the previous chunk
            overlap_words = current_chunk.split(" ")[-int(overlap/6):]
            current_chunk = " ".join(overlap_words) + " " + word + " "
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_and_chunk_all(base_dir="data"):
    """
    Scans data directory, extracts text page-by-page, and applies layout-aware chunking.
    """
    chunks_dataset = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' does not exist.")
        return chunks_dataset
    
    for course_code in os.listdir(base_dir):
        course_path = os.path.join(base_dir, course_code)
        
        if os.path.isdir(course_path):

            for filename in os.listdir(course_path):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(course_path, filename)
                    
                    try:
                        with pdfplumber.open(pdf_path) as pdf:
                            for page_num, page in enumerate(pdf.pages, start=1):
                                text = page.extract_text()
                                if not text:
                                    continue
                                    
  
                                page_chunks = chunk_text_recursively(text)
                                
                                for chunk in page_chunks:
                                    chunks_dataset.append({
                                        "text": chunk,
                                        "metadata": {
                                            "course": course_code,
                                            "filename": filename,
                                            "page": page_num
                                        }
                                    })
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    MAX_TOTAL_CHUNKS = 1000
    if len(chunks_dataset) > MAX_TOTAL_CHUNKS:
        chunks_dataset = chunks_dataset[:MAX_TOTAL_CHUNKS]

    return chunks_dataset