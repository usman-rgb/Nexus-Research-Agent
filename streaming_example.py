from test_llm import get_llm
import json

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
                if "text" in item:
                    extracted.append(str(item["text"]))
                else:
                    try:
                        extracted.append(json.dumps(item))
                    except Exception:
                        extracted.append(str(item))
            else:
                extracted.append(str(item))
        return "".join(extracted)
    elif isinstance(content, dict):
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    else:
        return str(content) if content else ""

def main():
    print("=" * 60)
    print(" Console Streaming Example with Gemini 3.6 Flash")
    print("=" * 60)

    # get_llm will return ChatGoogleGenerativeAI initialized with streaming=True 
    # and the gemini-3.5-flash model as we configured.
    llm = get_llm()
    if not llm:
        print("Failed to initialize LLM. Please check your .env file.")
        return

    prompt = "Explain quantum computing in exactly two short paragraphs."
    print(f"\n[USER PROMPT]: {prompt}\n")
    print("-" * 60)
    print("[LLM RESPONSE (Streaming)]:")
    
    # By using llm.stream(), we get the response token-by-token.
    try:
        for chunk in llm.stream(prompt):
            # Extract text safely using robust helper
            text = extract_safe_text(chunk.content)
                
            # end="" ensures chunks are printed side-by-side.
            # flush=True forces the terminal to display the text immediately.
            print(text, end="", flush=True)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during streaming: {e}")
        
    print("\n\n" + "-" * 60)
    print("[SUCCESS] Streaming finished!")

if __name__ == "__main__":
    main()
