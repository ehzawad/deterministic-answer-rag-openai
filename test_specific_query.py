#!/usr/bin/env python3
"""
Test specific query that should work but isn't
"""

from faq_service import faq_service

def test_specific():
    query = "পেরোল অ্যাকাউন্টের সুদের হার কত?"
    
    print(f"🔍 Testing: {query}")
    print("=" * 60)
    
    result = faq_service.answer_query(query, debug=True)
    
    if result["found"]:
        print(f"✅ MATCH FOUND!")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Matched: {result['matched_question']}")
        print(f"   Answer: {result['answer'][:150]}...")
    else:
        print(f"❌ NO MATCH")
        print(f"   Best score: {result.get('confidence', 0):.1%}")
    
    # Show ALL candidates
    print(f"\n🐛 ALL candidates ({len(result.get('candidates', []))}):")
    for i, candidate in enumerate(result.get('candidates', [])[:10], 1):
        score = candidate['score']
        collection = candidate.get('collection', 'unknown')
        question = candidate['question']
        # Look for the exact semantic match
        if 'ইন্টারসেট রেট' in question and 'পেরোল' in question:
            print(f"   {i}. ⭐ {question} (Score: {score:.3f}, Collection: {collection}) ⭐")
        else:
            print(f"   {i}. {question[:70]}... (Score: {score:.3f}, Collection: {collection})")

if __name__ == "__main__":
    test_specific()