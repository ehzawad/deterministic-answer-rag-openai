#!/usr/bin/env python3
"""
Interactive CLI for Bengali FAQ System
Session-based interface for real-time question answering
"""

import sys
from faq_service import faq_service

def main():
    """Main interactive session"""
    try:
        print("\n🚀 Bengali FAQ System - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("- Type your question in Bengali or English")
        print("- 'debug on/off' to toggle debug mode") 
        print("- 'stats' to see system statistics")
        print("- 'exit' to quit")
        print("=" * 50)
        
        # Check if service is initialized
        if not faq_service.initialized:
            print("\n⚠️  WARNING: FAQ Service is not initialized!")
            print("Please check your configuration and FAQ data files.")
            return
        
        # Show system stats on startup
        stats = faq_service.get_system_stats()
        print(f"\n✅ FAQ Service Ready!")
        if stats.get('test_mode', False):
            print("⚠️  Running in TEST MODE (no embeddings)")
        print(f"📊 Collections: {stats.get('total_collections', 0)}")
        for coll_name, coll_info in stats.get('collections', {}).items():
            print(f"   • {coll_name}: {coll_info['count']} entries")
        
        debug_mode = False
        
        while True:
            try:
                # Get user input
                prompt = f"\n{'🔍' if not debug_mode else '🐛'} Enter your query: "
                query = input(prompt).strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                elif query.lower() == 'debug on':
                    debug_mode = True
                    print("🐛 Debug mode enabled.")
                    continue
                elif query.lower() == 'debug off':
                    debug_mode = False
                    print("🔍 Debug mode disabled.")
                    continue
                elif query.lower() == 'stats':
                    stats = faq_service.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    if stats.get('test_mode', False):
                        print("Mode: TEST MODE (no embeddings)")
                    else:
                        print("Mode: Full embedding mode")
                    print(f"Collections: {stats.get('total_collections', 0)}")
                    for coll_name, coll_info in stats.get('collections', {}).items():
                        print(f"  • {coll_name}: {coll_info['count']} entries")
                    continue
                
                # Process the query
                result = faq_service.answer_query(query, debug=debug_mode)
                
                # Display results
                print("\n" + "="*50)
                if result["found"]:
                    print(f"✅ MATCH FOUND (Confidence: {result['confidence']:.1%})")
                    print(f"📁 Source: {result['source']}")
                    if 'collection' in result:
                        print(f"🗂️  Collection: {result['collection']}")
                    print(f"❓ Question: {result['matched_question']}")
                    print(f"💬 Answer:\n{result['answer']}")
                    
                    if debug_mode:
                        print(f"\n🐛 DEBUG INFO:")
                        print(f"   Detected Collections: {result.get('detected_collections', [])}")
                        print(f"   Confidence Threshold: {result.get('threshold', 'N/A')}")
                        if 'candidates' in result:
                            print(f"   Top Candidates:")
                            for i, candidate in enumerate(result['candidates'][:3], 1):
                                print(f"     {i}. {candidate['question'][:80]}... (Score: {candidate['score']:.3f})")
                else:
                    print(f"❌ NO MATCH FOUND")
                    if 'confidence' in result:
                        print(f"📊 Best candidate score: {result['confidence']:.1%}")
                    print(f"💬 {result.get('message', 'No answer available.')}")
                    
                    if debug_mode and 'candidates' in result:
                        print(f"\n🐛 DEBUG INFO - Top candidates:")
                        for i, candidate in enumerate(result['candidates'][:3], 1):
                            print(f"   {i}. {candidate['question'][:80]}... (Score: {candidate['score']:.3f})")
                
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"\n💥 System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 