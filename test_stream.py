import asyncio
from app import get_cached_agent

async def main():
    agent = get_cached_agent()
    messages = [("user", "Give me a quick 1-sentence summary of ML.")]
    
    print("Starting stream...")
    async for chunk, metadata in agent.astream({"messages": messages}, stream_mode="messages"):
        chunk_type = getattr(chunk, 'type', '') or ''
        is_ai = chunk_type in ("ai", "AIMessageChunk", "AIMessage") or "AI" in type(chunk).__name__
        
        print(f"\n[CHUNK_TYPE]: {chunk_type} | [IS_AI]: {is_ai} | [CLASS]: {type(chunk).__name__}")
        print(f"[CONTENT]: {repr(chunk.content)}")
        
if __name__ == "__main__":
    asyncio.run(main())
