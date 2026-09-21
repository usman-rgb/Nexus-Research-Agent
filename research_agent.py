import os
from test_llm import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent

def main():
    # 1. LLM Load Karein
    llm = get_llm()
    if not llm:
        print("LLM initialize nahi ho saka. Apni API keys check karein.")
        return

    # 2. Tools Setup Karein
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=800, load_max_docs=3, load_all_available_meta=False)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
    ddg_tool = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)
    
    # Custom tool wrapper to truncate DuckDuckGo output and save tokens/speed
    from langchain.tools import tool
    @tool
    def concise_search(query: str) -> str:
        """Search the web using DuckDuckGo but return concise truncated results."""
        result = ddg_tool.invoke(query)
        return result[:1000] + "... [Truncated for brevity]" if len(result) > 1000 else result
    
    tools = [arxiv_tool, concise_search]

    # 3. System Prompt Tayyar Karein
    system_prompt = """You are an 'Expert Research Assistant', capable of digging deep into complex topics. 
Your goal is to synthesize information from multiple sources to provide accurate, comprehensive, and well-structured answers.
For academic, scientific, or highly technical papers, always prioritize using Arxiv. For recent news, general facts, or things not found on Arxiv, use DuckDuckGo.
Always provide clear explanations and cite where your information came from."""

    # 4. Agent banayein LangGraph ka use karke
    # create_react_agent internally ek loop (graph) banata hai LLM aur tools ke darmiyan
    agent = create_react_agent(llm, tools)

    # 5. Agent ko Test Karein
    print("Agent is ready. Starting research...\n")
    query = "What is the QLoRA (Quantized Low-Rank Adaptation) technique? Find its explanation and who introduced it."
    
    print(f"Query: {query}\n")
    
    # LangGraph agent .invoke() ke liye "messages" key accept karta hai
    messages = [
        ("system", system_prompt),
        ("user", query)
    ]
    response = agent.invoke({"messages": messages})
    print("\n" + "="*50)
    print("FINAL RESEARCH RESULT:")
    print("="*50)
    # Agent ki aakhri message ka content print karein
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
