import os
from test_llm import get_llm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DB_DIR = "./chroma_db"

def get_embeddings():
    """Returns embeddings using GoogleGenerativeAIEmbeddings with a fallback to HuggingFace."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
            return GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)
        except Exception as e:
            print(f"Error initializing Google embeddings, falling back to HuggingFace: {e}")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found. Falling back to HuggingFaceEmbeddings.")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_split_pdf(file_path: str):
    """
    Loads a PDF file and splits it into manageable text chunks.
    
    Args:
        file_path (str): The local path to the PDF file.
        
    Returns:
        list: A list of Document objects containing the split text chunks.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The PDF file at {file_path} was not found.")
            
        print(f"Loading PDF from {file_path}...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            print("Warning: The loaded PDF appears to be empty.")
            return []
            
        print(f"Successfully loaded {len(documents)} page(s) from PDF.")
        
        # Split the document into chunks
        print("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Successfully split the document into {len(chunks)} chunk(s).")
        
        return chunks
        
    except Exception as e:
        print(f"An error occurred while loading and splitting the PDF: {e}")
        # Re-raise the exception if the caller wants to handle it, or return an empty list
        # We will return an empty list here to ensure the program doesn't crash unexpectedly, 
        # but the error is logged. Alternatively, raising it is also good practice.
        raise

def create_vector_store(chunks):
    """
    Takes text chunks and stores them in a local Chroma vector database.
    
    Args:
        chunks (list): A list of LangChain Document objects.
        
    Returns:
        Chroma: The initialized Chroma vector store.
    """
    try:
        from langchain_chroma import Chroma
        import chromadb
        
        print(f"Creating vector store with {len(chunks)} chunks...")
        embeddings = get_embeddings()

        # Check and handle dimension mismatch with existing collection
        if os.path.exists(CHROMA_DB_DIR):
            try:
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                collections = client.list_collections()
                for col in collections:
                    if col.name == "langchain":
                        sample = col.peek(1)
                        if sample and sample.get("embeddings") is not None and len(sample["embeddings"]) > 0:
                            existing_dim = len(sample["embeddings"][0])
                            test_dim = len(embeddings.embed_query("dimension test"))
                            if existing_dim != test_dim:
                                print(f"Notice: Existing collection dimension ({existing_dim}) differs from current embeddings ({test_dim}). Resetting collection...")
                                client.delete_collection("langchain")
            except Exception as dim_err:
                print(f"Notice during collection dimension check: {dim_err}")
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("Vector store created successfully in directory:", CHROMA_DB_DIR)
        return vector_store
    except Exception as e:
        print(f"An error occurred while creating the vector store: {e}")
        raise

def get_retriever():
    """
    Returns a LangChain retriever interface for the local Chroma vector store.
    Fetches the top 3 most relevant chunks based on semantic similarity.
    
    Returns:
        VectorStoreRetriever: The configured LangChain retriever.
    """
    try:
        from langchain_chroma import Chroma
        
        print("Initializing retriever...")
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("Retriever initialized successfully (k=3).")
        return retriever
    except Exception as e:
        print(f"An error occurred while getting the retriever: {e}")
        raise

def extract_pdf_topics(chunks):
    """
    Extracts 3-5 main topics from the provided PDF chunks using the LLM.
    Includes automatic retry on rate limit errors.
    
    Args:
        chunks (list): A list of LangChain Document objects.
        
    Returns:
        list: A list of strings representing the extracted topics.
    """
    import time
    
    if not chunks:
        return []
        
    # Combine the text of the first 3 chunks (to save tokens and speed up)
    text_to_analyze = "\n".join([doc.page_content for doc in chunks[:3]])
    
    prompt = f"""Analyze the following text extracted from a PDF and identify exactly 3 to 5 main topics or specific key concepts discussed in it. 
These topics should be short, concise, and phrased as research queries that a user might ask an AI assistant.
Do not include any introductory or concluding text, bullet points, or numbering. 
Just return a simple comma-separated list of the topics.

TEXT:
{text_to_analyze[:4000]}"""

    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            llm = get_llm()
            if not llm:
                return []
                
            response = llm.invoke(prompt)
            
            # Extract text safely
            if isinstance(response.content, str):
                output_text = response.content
            elif isinstance(response.content, list):
                output_text = "".join(
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in response.content
                )
            else:
                output_text = str(response.content)
                
            # Parse the comma-separated string into a list
            topics = [t.strip() for t in output_text.split(',') if t.strip()]
            
            # Fallback if the LLM didn't format correctly
            if not topics or len(topics) < 2:
                # Maybe they used newlines
                topics = [t.strip('-* \n') for t in output_text.split('\n') if t.strip('-* \n')]
                
            return topics[:5] # Max 5 topics
            
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                if attempt < max_retries:
                    wait_time = 40  # seconds
                    print(f"[RATE LIMIT] API quota exceeded. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[RATE LIMIT] Still rate-limited after {max_retries} retries. Generating fallback topics from text...")
                    return _generate_fallback_topics(chunks)
            else:
                print(f"Error extracting topics: {e}")
                return _generate_fallback_topics(chunks)
    
    return []

def _generate_fallback_topics(chunks):
    """
    Generates simple fallback topics from chunk content without using the LLM.
    Uses basic text analysis to extract meaningful phrases.
    """
    if not chunks:
        return []
    
    # Combine all text
    full_text = " ".join([doc.page_content for doc in chunks[:5]])
    
    # Simple keyword extraction: find capitalized multi-word phrases and common section headers
    import re
    
    # Find phrases that look like topics (title-cased words, 2-5 words long)
    # Look for patterns after common heading indicators
    heading_patterns = re.findall(r'(?:^|\n)#+\s*(.+?)(?:\n|$)', full_text)
    bold_patterns = re.findall(r'\*\*(.+?)\*\*', full_text)
    
    candidates = heading_patterns + bold_patterns
    
    # Clean and deduplicate
    topics = []
    seen = set()
    for c in candidates:
        c = c.strip().strip('#*: ')
        c_lower = c.lower()
        if 3 < len(c) < 80 and c_lower not in seen:
            seen.add(c_lower)
            topics.append(c)
        if len(topics) >= 5:
            break
    
    # If still not enough, extract first few sentences as topic hints
    if len(topics) < 3:
        sentences = re.split(r'[.!?]\s+', full_text[:2000])
        for s in sentences:
            s = s.strip()
            if 10 < len(s) < 80 and s.lower() not in seen:
                seen.add(s.lower())
                topics.append(s)
            if len(topics) >= 4:
                break
    
    return topics[:5]

