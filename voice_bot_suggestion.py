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
        "ইয়াকিন টার্ম ডিপোজিট একাউন্ট খুলতে ডকুমেন্ট কি লাগবে",
        "ইয়াকিন এসএমই একাউন্ট খুলতে কি কি প্রয়োজন",
        "ইয়াকিন বানাত জেনারেল একাউন্ট খুলতে কত টাকা লাগে",
        "ইয়াকিন সেভিংস একাউন্টের সুবিধা এবং প্রফিট রেট জানতে চাই",
        "ইয়াকিন ফ্লেক্সিবল সেভিংস স্কিমের মেয়াদ ও সুবিধা কি",
        "ইয়াকিন লাখপতি স্কিমের বিস্তারিত তথ্য জানতে চাই",
        "ইয়াকিন কোটিপতি স্কিমের বিস্তারিত তথ্য জানতে চাই",
        "ইয়াকিন বান্নাত প্রিমিয়াম সেভিংস একাউন্টের সুবিধা বলুন",
        "ইয়াকিন ভিসা প্ল্যাটিনাম কার্ডের সুবিধা বলুন",
        "ইসলামিক সিগনেচার কার্ডের সুবিধা বলুন",
        "ইসলামিক কারেন্ট একাউন্ট খুলতে ন্যূনতম ব্যালেন্স কত",
        "ইয়াকিন মান্থলি সেভিংস স্কিম খুলতে ন্যূনতম কিস্তি কত",
        "মেয়েদের জন্য ইয়াকিনের বিশেষ একাউন্ট বা স্কিম আছে কি",
        "ইয়াকিন বানাত ডিপিএস স্কিমের আইএসআর কত",
        "ইয়াকিন বান্নাত টার্ম ডিপোজিটের ইনকাম শেয়ারিং রেশিও জানতে চাই",
        "ইয়াকিন জুনিয়র একাউন্টের বয়সসীমা ও সুবিধাগুলো কি কি",
        "ইসলামিক ফিক্সড ডিপোজিটের মেয়াদ এবং প্রফিট রেট কত",
        "ইয়াকিন আরডি রিকারিং ডিপোজিট প্রোডাক্টের বিস্তারিত বলুন",
        "ইয়াকিন রিকারিং ডিপোজিট প্রোডাক্টের",
        "ইসলামিক ডিপিএস এর প্রফিট রেট এবং আইএসআর সম্পর্কে বলুন",
        "ইয়াকিন মান্থলি সেভিংস স্কিম খুলতে ন্যূনতম কিস্তি কত",
        "ইয়াকিন ফ্লেক্সিবল সেভিংস স্কিমের মেয়াদ ও সুবিধা কি",
        "ইয়াকিন লাখপতি স্কিমের বিস্তারিত তথ্য জানতে চাই",
        "ইয়াকিন কোটিপতি স্কিমের বিস্তারিত তথ্য জানতে চাই",
        "ইয়াকিন পেনশন স্কিমের আইএসআর কত",
        "ইসলামিক হোম লোনের রেন্টাল রেট এবং আবেদন পদ্ধতি বলুন",
        "ইয়াকিন অটো লোনের প্রফিট রেট ও এলিজিবিলিটি জানতে চাই",
        "ইসলামিক পার্সোনাল ফাইন্যান্সের শর্তাবলী এবং রেট কত",
        "ইসলামিক এসএমই লোন বা ইনভেস্টমেন্ট এর প্রফিট রেট জানতে চাই",
        "ইসলামিক রিফাইন্যান্স পেতে কি করতে হবে",
        "ইসলামিক প্রিফাইন্যান্স পেতে কি করতে হবে",
        "ইসলামিক ক্রেডিট কার্ড চার্জ কত",
        "ইয়াকিন উজরা কার্ডের চার্জ কত",
        "ইয়াকিন ভিসা প্ল্যাটিনাম কার্ডের সুবিধা বলুন",
        "ইসলামিক সিগনেচার কার্ডের সুবিধা বলুন",
        "ইয়াকিন পেরোল একাউন্টে ডেবিট কার্ড ও চেক বই চার্জ আছে কি",
        "ইসলামিক পেরোল একাউন্ট অনলাইনে খুলতে পারবো কিনা বলুন",
        "আমি প্রবাসী, ইয়াকিন একাউন্টে রেমিট্যান্স পাঠানো যাবে কিনা জানতে চাই",
        "আপনাদের শরিয়া সুপারভাইজার ই কমিটিতে কারা আছেন",
        "আপনাদের শরিয়া সুপারভাইজরি কমিটিতে কারা আছেন",
        "এমটিবি ইসলামিক ব্যাংকিং সেবা সম্পর্কে জানতে চাই",
        "ইয়াকিন ব্যাংকিং উইং বা উইন্ডো কোথায় পাবো",
        "এফটিপি ইয়া কিনে কি কি প্রোডাক্ট আছে"
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