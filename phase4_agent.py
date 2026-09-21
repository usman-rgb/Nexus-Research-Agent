import os
from test_llm import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent

def main():
    print("--- Phase 4: LangGraph ReAct Loop ---")
    # 1. LLM Load
    llm = get_llm()
    if not llm:
        print("[ERROR] LLM load nahi hua.")
        return

    # 2. Tools Setup
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

    # 3. System Prompt for Markdown formatting
    system_prompt = "You are an Expert Research Assistant. Always provide your Final Answer in well-structured Markdown format with clear headings and bullet points. Cite your sources. Ensure your output is neat and readable."

    # 4. Initialize LangGraph ReAct Agent
    # 4. Initialize LangGraph ReAct Agent without modifier (since signature changed in v1.0)
    agent = create_react_agent(llm, tools)

    # 5. Query Pass Karna
    query = "What are Physics-informed neural networks (PINNs)? Briefly explain their core concept and give one practical application."
    print(f"\n[USER QUERY]: {query}\n")
    print("="*60)
    print("[AGENT'S THOUGHT PROCESS & ACTIONS (STREAMING)]:")
    print("="*60)
    
    # .stream() ka use taake agent ki har soch aur action step-by-step nazar aaye
    # Hum system prompt ko messages list ke andar hi de rahe hain
    final_output = ""
    messages = [
        ("system", system_prompt),
        ("user", query)
    ]
    
    for step in agent.stream({"messages": messages}, stream_mode="values"):
        message = step["messages"][-1]
        
        # Agar yeh AI/Agent ka message hai (Thought or Action)
        if message.type == "ai":
            if message.tool_calls:
                print(f"[THOUGHT/DECISION]: I need to use tools to answer this.")
                for tc in message.tool_calls:
                    print(f"[ACTION]: Using Tool '{tc['name']}' with input: {tc['args']}")
            elif message.content:
                final_output = message.content
        
        # Agar yeh Tool ka observation/result hai
        elif message.type == "tool":
            print(f"[OBSERVATION from '{message.name}']: \n{message.content[:250]}... [truncated]")
            print("-" * 40)

    print("\n" + "="*60)
    print("[FINAL MARKDOWN OUTPUT]:")
    print("="*60)
    print(final_output)

if __name__ == "__main__":
    main()
