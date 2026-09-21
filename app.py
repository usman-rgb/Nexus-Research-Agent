import os
import json
import uvicorn
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Agent imports
from test_llm import get_llm
import arxiv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from rag_utils import get_retriever

app = FastAPI()

# Mount static files (for style.css, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 Templates (for index.html)
templates = Jinja2Templates(directory="templates")

# Request Model for JSON body validation
class ResearchQuery(BaseModel):
    query: str

def extract_safe_text(content):
    """Robustly extract text from complex nested LangGraph chunks (lists, dicts, etc)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict):
                # New Gemini format: {'type': 'text', 'text': '...', 'extras': {...}}
                if "text" in item and item["text"]:
                    extracted.append(str(item["text"]))
                # Skip items that only have metadata/signatures/extras with no text
            else:
                extracted.append(str(item))
        return "".join(extracted)
    elif isinstance(content, dict):
        if "text" in content and content["text"]:
            return str(content["text"])
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    else:
        return str(content) if content else ""

def get_agent():
    llm = get_llm()
    if not llm:
        return None

    # CUSTOM ARXIV TOOL: Uses arxiv library directly with max_results=3 and low delay
    @tool("arxiv")
    def error_safe_arxiv(query: str) -> str:
        """Search academic papers on Arxiv. Use for scientific/technical research papers."""
        try:
            # Low delay and 1 retry for maximum speed
            client = arxiv.Client(page_size=3, delay_seconds=1.0, num_retries=1)
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for paper in client.results(search):
                results.append(
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                    f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                    f"Summary: {paper.summary[:500]}\n"
                    f"URL: {paper.entry_id}\n"
                )
            if results:
                return "\n---\n".join(results)
            return "No results found on Arxiv for this query."
        except Exception as e:
            return 'Arxiv search failed (API Error). Proceed to use DuckDuckGo web search instead.'

    # CUSTOM DUCKDUCKGO TOOL: Error-safe web search with truncation
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    ddg_tool = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)

    @tool
    def concise_search(query: str) -> str:
        """Search the web using DuckDuckGo for recent news, facts, or general information."""
        try:
            result = ddg_tool.invoke(query)
            return result[:1500] + "... [Truncated]" if len(result) > 1500 else result
        except Exception as e:
            return f"Web search failed: {str(e)}. Please provide your best answer based on your training data."

    @tool("search_local_pdf")
    def search_local_pdf(query: str) -> str:
        """Search local uploaded PDF documents for relevant context. Use when the query relates to uploaded files or user documents."""
        # Fast path: Check if any PDFs exist before doing heavy vector store operations
        uploads_dir = "uploads"
        chroma_dir = "./chroma_db"
        has_uploads = os.path.exists(uploads_dir) and any(f.lower().endswith('.pdf') for f in os.listdir(uploads_dir))
        has_chroma = os.path.exists(chroma_dir) and len(os.listdir(chroma_dir)) > 0
        
        if not has_uploads and not has_chroma:
            return "No local PDF documents have been uploaded yet. Please use concise_search or arxiv for this query."

        try:
            retriever = get_retriever()
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant context found in local PDF."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"--- Chunk {i+1} ---\n{doc.page_content}")
            return "\n".join(results)
        except Exception as e:
            return f"Error searching local PDF: {str(e)}"

    tools = [search_local_pdf, error_safe_arxiv, concise_search]
    agent = create_react_agent(llm, tools)
    return agent

# Cache the agent at module level so it's not recreated on every request (SPEED FIX)
_cached_agent = None

def get_cached_agent():
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = get_agent()
    return _cached_agent

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"success": False, "message": "Only PDF files are supported."}
    
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        
        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the PDF for RAG
        from rag_utils import load_and_split_pdf, create_vector_store, extract_pdf_topics
        chunks = load_and_split_pdf(file_path)
        if not chunks:
            return {"success": False, "message": "Could not extract text from PDF."}
            
        # Store in vector DB
        create_vector_store(chunks)
        
        # Extract dynamic topics
        topics = extract_pdf_topics(chunks)
        
        msg = f"Successfully processed {len(chunks)} chunks from {file.filename}."
        if not topics:
            msg += " (Warning: Topic extraction skipped due to API Rate Limit. Please wait ~30 seconds before searching)."
            
        return {
            "success": True, 
            "message": msg,
            "topics": topics
        }
    except Exception as e:
        return {"success": False, "message": f"Error processing PDF: {str(e)}"}

@app.post("/research_stream")
async def research_stream(data: ResearchQuery):
    if not data.query:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "content": "No query provided"}) + "\n"]), 
            status_code=400, 
            media_type='application/x-ndjson'
        )
        
    agent = get_cached_agent()
    if not agent:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "content": "Failed to initialize LLM."}) + "\n"]), 
            status_code=500, 
            media_type='application/x-ndjson'
        )
        
    system_prompt = """You are an 'Expert Research Assistant', capable of digging deep into complex topics. 
Your goal is to synthesize information from multiple sources to provide accurate, comprehensive, and well-structured answers.

TOOL USAGE GUIDELINES:
1. If the user query specifically asks about uploaded PDFs, local documents, or user files, use the `search_local_pdf` tool.
2. For general questions, concepts, science, code, or real-world topics, prioritize `concise_search` (web search) or `arxiv` (for research papers).
3. Only call tools when external or specific information is needed. If you already have full knowledge to provide a high-quality answer, you can synthesize directly.

Always provide clear explanations.
CRITICAL: You MUST include a section titled "### Important Points" summarizing key takeaways.
CRITICAL: You MUST include sources and references. At the end of your response, provide a "### References" section listing all the sources you used.
Your final answer must be written entirely in Markdown format."""

    messages = [
        ("system", system_prompt),
        ("user", data.query)
    ]
    
    # Fast recursion limit (prevents infinite tool-calling loops)
    config = {"recursion_limit": 15}
    
    async def generate():
        yield json.dumps({"type": "log", "content": "🚀 Initializing Async Research Agent (FastAPI)..."}) + "\n"
        try:
            previous_tool_calls = set()
            
            async for event in agent.astream({"messages": messages}, stream_mode="values", config=config):
                # We examine the latest message in the state
                if "messages" not in event or not event["messages"]:
                    continue
                    
                last_msg = event["messages"][-1]
                
                if last_msg.type == "tool":
                    yield json.dumps({"type": "log", "content": f"✅ Observation received from: {last_msg.name}"}) + "\n"
                
                elif last_msg.type == "ai":
                    # Check for tool calls
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            tc_id = tc.get("id")
                            if tc_id and tc_id not in previous_tool_calls:
                                previous_tool_calls.add(tc_id)
                                yield json.dumps({"type": "log", "content": f"🛠️ Calling tool: {tc['name']}..."}) + "\n"
                    
                    # If this is the final AI response (no tool calls left)
                    elif not getattr(last_msg, "tool_calls", None) and last_msg.content:
                        content = extract_safe_text(last_msg.content)
                        if content:
                            yield json.dumps({"type": "final_answer", "content": content}) + "\n"

        except Exception as e:
            err_msg = str(e)
            yield json.dumps({"type": "log", "content": f"❌ Error encountered: {err_msg}"}) + "\n"
            yield json.dumps({"type": "error", "content": f"An error occurred: {err_msg}"}) + "\n"
            
        yield json.dumps({"type": "log", "content": "🏁 Research completed."}) + "\n"

    # StreamingResponse is natively compatible with async generators in FastAPI
    return StreamingResponse(generate(), media_type='application/x-ndjson')

if __name__ == "__main__":
    print("Starting FastAPI application via Uvicorn...")
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
