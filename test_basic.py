#!/usr/bin/env python3
"""Basic test script for the Bengali FAQ system"""

from faq_service import faq_service
import json

def test_basic_query():
    """Test a basic Bengali query"""
    test_query = 'পেরোল একাউন্টের সুবিধা কি?'
    print(f'Testing query: {test_query}')
    
    try:
        result = faq_service.answer_query(test_query, debug=False)
        print('✅ Query successful!')
        print('Result:')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        print(f'❌ Query failed: {e}')
        return False

def test_system_stats():
    """Test system statistics"""
    try:
        stats = faq_service.get_system_stats()
        print('\n📊 System Stats:')
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        print(f'❌ Stats failed: {e}')
        return False

if __name__ == "__main__":
    print("🚀 Bengali FAQ System - Basic Test")
    print("=" * 50)
    
    # Test basic query
    success1 = test_basic_query()
    
    # Test system stats
    success2 = test_system_stats()
    
    if success1 and success2:
        print("\n✅ All basic tests passed!")
    else:
        print("\n❌ Some tests failed!") 