# NCERT RAG Chatbot - Documentation

## Overview
The NCERT RAG (Retrieval-Augmented Generation) chatbot is designed to provide responses based on NCERT science textbooks (Classes 6-10). The project follows a pipeline involving data scraping, text extraction, cleaning, chunking, embedding, FAISS indexing, and retrieval-based answering.

## 1. Scraping NCERT Website
**File:** `scraper.py`
This script downloads NCERT science books in PDF format, chapter-wise, using Selenium.

### Key Features:
- Uses Selenium WebDriver to automate downloads. 
- Saves PDFs in the `NCERT_downloads/` folder. 
- Extracts complete textbooks for classes 6 to 10. 

### Usage:
```bash
python scraper.py
```

### Implementation Details:
- WebDriver navigates to the NCERT website. 
- Locates and clicks the "Download complete book" button. 
- Waits for the download to complete. 
- Extracts and organizes downloaded books. 

---

## 2. Extracting Text from PDFs
**File:** `extract.py`
Extracts text from the downloaded PDFs using PyMuPDF (fitz).

### Key Features:
- Reads PDF files and extracts text page-wise. 
- Saves extracted text in the `NCERT_downloads/extracted_texts/` directory. 

### Usage:
```bash
python extract.py
```

### Implementation Details:
```python
pdf_document = fitz.open(pdf_path)
text = "".join([page.get_text() for page in pdf_document])
```

---

## 3. Cleaning Extracted Text
**File:** `filter.py`
Removes unwanted elements like headers, footers, and page numbers from extracted text.

### Key Features:
- Removes page numbers, chapter headers, and special characters. 
- Saves cleaned text in the `cleaned_texts/` directory. 

### Usage:
```bash
python filter.py
```

### Implementation Details:
```python
text = re.sub(r'Page\s*\d+', '', text)  # Remove "Page 1", "Page 2"
text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
```

---

## 4. Chunking Text using LangChain
**File:** `chunking.py`
Splits cleaned text into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

### Key Features:
- Chunks of 1000 characters with an overlap of 100 characters. 
- Saves the chunks in `chunks_output/`. 

### Usage:
```bash
python chunking.py
```

### Implementation Details:
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(text)
```

---

## 5. Creating Embeddings & Indexing with FAISS
**File:** `main.py`
Processes text chunks into vector embeddings and stores them in FAISS.

### Key Features:
- Converts text chunks into embeddings using OpenAI's `text-embedding-ada-002` model. 
- Stores the vectors in a persistent FAISS index. 

### Usage:
```bash
python main.py
```

### Implementation Details:
```python
import openai

def generate_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"])
```

---

## 6. Implementing the RAG Chatbot
**File:** `app.py`
Provides a Streamlit UI for user queries, retrieves relevant documents, and generates responses.

### Key Features:
- Streamlit-based UI for easy interaction.
- Retrieves most relevant chunks from the FAISS index.
- Uses an LLM to generate responses. 

### Usage:
```bash
streamlit run app.py
```

### Implementation Details:
```python
import streamlit as st

st.title("NCERT RAG Chatbot")
query = st.text_input("Ask a question about NCERT science textbooks")
if query:
    query_embedding = generate_embedding(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k=5)
    st.write("Most relevant text chunks:", indices)
```

---

## 7. Testing the RAG Chatbot
**File:** `test_chatbot.py`
Ensures that:
- The chatbot correctly retrieves relevant responses. 
- FAISS indexing and retrieval are working as expected. 

### Usage:

#### Example Test Cases:
```python
# Add test cases for chatbot response validation
