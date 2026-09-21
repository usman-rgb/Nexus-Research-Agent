import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file located in the script's directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

import langchain
from langchain_community.cache import InMemoryCache, SQLiteCache

# ==============================================================================
# GLOBAL Caching Setup (Saves API calls, money, and time during testing)
# ==============================================================================

# OPTION 1: InMemoryCache (Fastest, but clears when script/server restarts)
langchain.llm_cache = InMemoryCache()

# OPTION 2: SQLiteCache (Persistent across restarts, saves to a local database file)
# To use this, comment out Option 1 and uncomment the line below:
# langchain.llm_cache = SQLiteCache(database_path=".langchain_cache.db")

def get_llm():
    """Initializes and returns the LLM instance based on the configured provider (Gemini or OpenAI)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("[ERROR] GEMINI_API_KEY is not set.")
            print("Please add your Gemini API key to the '.env' file.")
            sys.exit(1)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
            print(f"[INFO] Initializing Gemini LLM ({model_name}) with fallback support...")
            
            # Primary fast model
            primary_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                streaming=True,
                max_retries=1
            )
            
            # Fallback models in case of high demand / 503 errors
            backup_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
            fallbacks = []
            for b_model in backup_models:
                if b_model != model_name:
                    fallbacks.append(
                        ChatGoogleGenerativeAI(
                            model=b_model,
                            google_api_key=api_key,
                            streaming=True,
                            max_retries=1
                        )
                    )
            
            if fallbacks:
                return primary_llm.with_fallbacks(fallbacks)
            return primary_llm
        except ImportError:
            print("[ERROR] 'langchain-google-genai' package is not installed.")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("[ERROR] OPENAI_API_KEY is not set.")
            print("Please add your OpenAI API key to the '.env' file.")
            sys.exit(1)

        try:
            from langchain_openai import ChatOpenAI
            print("[INFO] Initializing OpenAI LLM (gpt-4o-mini)...")
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key,
                temperature=0.3,
                streaming=True
            )
        except ImportError:
            print("[ERROR] 'langchain-openai' package is not installed.")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)

    else:
        print(f"[ERROR] Unknown LLM_PROVIDER: '{provider}'. Supported options are 'gemini' or 'openai'.")
        sys.exit(1)

def main():
    print("=" * 60)
    print(" Automated Research Agent - Phase 1: LLM Connection Test")
    print("=" * 60)

    # Initialize LLM
    llm = get_llm()

    # Simple test prompt
    test_prompt = "Hello! Please reply in one short sentence confirming that you are connected and ready to assist as an Automated Research Agent."
    print(f"\n[USER PROMPT]: {test_prompt}\n")

    try:
        # Invoke LLM
        response = llm.invoke(test_prompt)
        # Extract text content cleanly
        if isinstance(response.content, str):
            output_text = response.content
        elif isinstance(response.content, list):
            output_text = "".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in response.content
            )
        else:
            output_text = str(response.content)

        print("-" * 60)
        print("[LLM RESPONSE]:")
        print(output_text.strip())
        print("-" * 60)
        print("\n[SUCCESS] LLM connection tested successfully!")
    except Exception as e:
        print(f"\n[ERROR] An error occurred while calling the LLM:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
