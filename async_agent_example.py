import asyncio
from test_llm import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent

# Async main function
async def main():
    llm = get_llm()
    if not llm:
        print("LLM initialize nahi ho saka.")
        return

    # Tools Setup
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=800, load_max_docs=3, load_all_available_meta=False)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
    ddg_tool = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)
    
    from langchain.tools import tool
    @tool
    def concise_search(query: str) -> str:
        """Search the web using DuckDuckGo but return concise truncated results."""
        result = ddg_tool.invoke(query)
        return result[:1000] + "... [Truncated for brevity]" if len(result) > 1000 else result

    tools = [arxiv_tool, concise_search]

    # Initialize LangGraph Agent
    # Note: If you are using legacy AgentExecutor, you would pass max_iterations=3 to its constructor.
    # But since you are using LangGraph's create_react_agent, we handle limits in the invoke configuration.
    agent = create_react_agent(llm, tools)

    query = "What is QLoRA? Briefly explain."
    print(f"[USER]: {query}")
    print("=" * 60)
    
    messages = [("user", query)]
    
    # OPTIMIZATION 1 & 2: Async Invocation + Loop Limit
    # LangGraph uses 'recursion_limit' in the config to prevent infinite loops.
    # recursion_limit=4 means it will take max 4 steps (e.g. LLM -> Tool -> LLM -> Finish)
    config = {"recursion_limit": 12} 
    
    try:
        # Use .ainvoke() instead of .invoke()
        # This is non-blocking and perfect for async web servers (FastAPI, Flask 2.0+ async routes)
        response = await agent.ainvoke({"messages": messages}, config=config)
        
        print("\n[AGENT RESPONSE]:")
        print(response["messages"][-1].content)
        
    except Exception as e:
        # If recursion_limit is exceeded, LangGraph raises a GraphRecursionError
        print(f"\n[ERROR/TIMEOUT]: {e}")

if __name__ == "__main__":
    # Run the async loop
    asyncio.run(main())
