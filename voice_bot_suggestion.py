#!/usr/bin/env python3
"""
Temporary test file to demonstrate Voice Bot Suggestion feature
Shows how the system would work with partial/corrupted voice input
"""

from faq_service import faq_service

def test_voice_bot_suggestion(query: str):
    """Test voice bot suggestion logic"""
    print(f"🎙️  Voice Input: '{query}'")
    print("=" * 60)
    
    # Process the query normally
    result = faq_service.answer_query(query, debug=False)
    
    if result["found"] and result["confidence"] >= 0.9:
        # Normal successful match
        print(f"✅ DIRECT MATCH (Confidence: {result['confidence']:.1%})")
        print(f"💬 Answer: {result['answer']}")
    else:
        # Failed match - show voice bot suggestion
        print(f"❌ NO DIRECT MATCH (Best Score: {result.get('confidence', 0):.1%})")
        
        # Get candidates from debug mode to access candidates array
        debug_result = faq_service.answer_query(query, debug=True)
        candidates = debug_result.get('candidates', [])
        
        if candidates and len(candidates) > 0:
            best_candidate = candidates[0]
            suggested_question = best_candidate['question']
            
            print(f"🤖 Voice Bot Response:")
            print(f"   '{suggested_question} আপনি কি সেটা বোঝাতে চেয়েছিলেন?'")
            print(f"   (Based on {best_candidate['score']:.1%} similarity)")
            
            # Simulate user saying "Yes" and getting the answer
            print(f"\n👤 User: 'হ্যাঁ'")
            print(f"🤖 Bot Answer: {best_candidate['answer']}")
            
        else:
            print("🤖 Voice Bot Response:")
            print("   'দুঃখিত, আমি বুঝতে পারিনি। আবার বলুন।'")
    
    print("=" * 60)
    print()

def main():
    """Test the voice bot suggestion scenarios"""
    print("🎙️  VOICE BOT SUGGESTION TESTING")
    print("=" * 60)
    print("Testing scenarios with partial/corrupted voice input")
    print()
    
    # Test cases based on user's example
    test_cases = [
        # User's actual example - missing beginning
        "ষি লোন এ সর্বনিম্ন কত টাকা দেয়া",
        
        # User's actual example - missing ending  
        "কৃষি লোন এ সর্বনিম্ন",
        
        # User's actual example - complete (should work normally)
        "কৃষি লোন এ সর্বনিম্ন কত টাকা",
        
        # Additional test cases
        "এমটিবি রেগুলার একাউন্ট",  # Partial conventional banking
        "ইয়াকিন সেভিংস",            # Partial Islamic banking
        "কার্ড ব্লক",               # Very short query
    ]
    
    for query in test_cases:
        test_voice_bot_suggestion(query)
    
    print("🎯 VOICE BOT SUGGESTION LOGIC:")
    print("1. If confidence >= 90% → Direct answer")
    print("2. If confidence < 90% → Suggest candidates[0].question")
    print("3. Add 'আপনি কি সেটা বোঝাতে চেয়েছিলেন?' to suggestion")
    print("4. Wait for user confirmation (হ্যাঁ/না)")

if __name__ == "__main__":
    main() 