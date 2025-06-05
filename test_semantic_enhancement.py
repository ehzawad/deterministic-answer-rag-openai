#!/usr/bin/env python3
"""
Test script for Semantic Enhancement with gpt-4.1-nano
Tests the previously failed queries to validate semantic understanding improvements
"""

from faq_service import faq_service

def test_semantic_queries():
    """Test semantic understanding with key queries"""
    
    test_queries = [
        # Previously failed queries
        "অ্যাকাউন্ট খুলতে শাখায় যেতে হবে কি?",
        "পেরোল অ্যাকাউন্টের সুদের হার কত?",
        
        # Additional semantic test cases
        "ইয়াকিন একাউন্টের ইন্টারেস্ট রেট কত?",  # Islamic banking interest
        "এমটিবি রেগুলার অ্যাকাউন্টে কি চার্জ আছে?",  # Conventional banking charges
        "পেরোল কার্ডের ফি কেমন?",  # Card fees
    ]
    
    print("🧪 Testing Semantic Enhancement with gpt-4.1-nano")
    print("=" * 60)
    
    if not faq_service.initialized:
        print("❌ FAQ Service not initialized!")
        return
    
    stats = faq_service.get_system_stats()
    print(f"✅ FAQ Service Ready!")
    print(f"📊 Test mode: {faq_service.test_mode}")
    print(f"📊 Collections: {stats.get('total_collections', 0)}")
    print(f"📊 Total entries: {sum(c['count'] for c in stats.get('collections', {}).values())}")
    print()
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Test {i}: {query}")
        print("-" * 60)
        
        # Process the query with debug mode
        result = faq_service.answer_query(query, debug=True)
        
        test_result = {
            'query': query,
            'found': result["found"],
            'confidence': result.get("confidence", 0)
        }
        
        if result["found"]:
            print(f"✅ MATCH FOUND!")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Matched: {result['matched_question']}")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Source: {result['source']}")
            print(f"   Collection: {result.get('collection', 'unknown')}")
            
            test_result.update({
                'matched_question': result['matched_question'],
                'answer': result['answer'],
                'source': result['source'],
                'collection': result.get('collection', 'unknown')
            })
        else:
            print(f"❌ NO MATCH")
            print(f"   Best score: {result.get('confidence', 0):.1%}")
            print(f"   Message: {result.get('message', 'No message')}")
        
        # Show debug info for semantic analysis
        if 'candidates' in result and result['candidates']:
            print(f"\n🐛 Top 3 candidates:")
            for j, candidate in enumerate(result['candidates'][:3], 1):
                score = candidate['score']
                collection = candidate.get('collection', 'unknown')
                question = candidate['question'][:60]
                print(f"   {j}. {question}... (Score: {score:.3f}, Collection: {collection})")
        
        results.append(test_result)
        print("\n" + "=" * 60)
        print()
    
    # Summary
    total_queries = len(test_queries)
    successful_matches = sum(1 for r in results if r['found'])
    match_rate = (successful_matches / total_queries) * 100
    
    print("📊 SEMANTIC ENHANCEMENT TEST SUMMARY")
    print("=" * 60)
    print(f"Total queries: {total_queries}")
    print(f"Successful matches: {successful_matches}")
    print(f"Match rate: {match_rate:.1f}%")
    print(f"Average confidence: {sum(r['confidence'] for r in results if r['found']) / max(successful_matches, 1):.1%}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    test_semantic_queries()