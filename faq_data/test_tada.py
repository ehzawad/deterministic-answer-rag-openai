#!/usr/bin/env python3
"""
Quick test script for the fixed Bengali FAQ RAG system to verify NRB account queries
"""

import asyncio
from bengali_faq_rag import AsyncRAGSystem

async def main():
    print("Testing fixed Bengali FAQ system specifically for NRB queries...")
    rag = AsyncRAGSystem()
    
    # Initialize the system
    await rag.initialize_async()
    
    # Test with the query that was failing
    test_query = "আমি একজন প্রবাসী, আমি কি আপনার ব্যাংকে এন আর বি সেভিংস একাউন্ট খুলতে পারবো"
    
    print(f"\nTesting query: {test_query}")
    result = await rag.answer_query_async(test_query, debug_mode=True)
    
    print("\n" + "="*60)
    if result["found"]:
        print(f"Matched Question: {result.get('matched_question', 'N/A')}")
        print(f"Confidence Score: {result.get('confidence', 0):.4f}")
        print(f"Source: {result.get('source_file', 'N/A')}")
        print(f"Search Method: {result.get('search_method', 'unknown')}")
        print("\nAnswer:")
        print(result["answer"])
        
        # Show debug info
        if "direct_search_results" in result:
            print("\nDirect Search Results:")
            for i, r in enumerate(result["direct_search_results"]):
                print(f"{i+1}. Q: {r['question']}")
                print(f"   A: {r['answer']}")
                print(f"   Score: {r['calibrated_score']:.4f}")
    else:
        print("\nNo suitable answer found:")
        print(result.get("message", "Please rephrase your question."))
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
