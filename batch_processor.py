#!/usr/bin/env python3
"""
Batch processor for Bengali FAQ System
Reads queries from input file line by line and processes them
"""

import sys
import json
import argparse
from datetime import datetime
from faq_service import faq_service

def process_batch_file(input_file: str, output_file: str = None, debug: bool = False):
    """Process queries from input file line by line"""
    
    if not faq_service.initialized:
        print("❌ FAQ Service not initialized!")
        return False
    
    # Default output file name
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_results_{timestamp}.json"
    
    print(f"🔄 Processing batch file: {input_file}")
    print(f"📄 Output file: {output_file}")
    
    # Check if input file exists
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return False
    
    if not queries:
        print("❌ No queries found in input file")
        return False
    
    print(f"📊 Found {len(queries)} queries to process")
    
    # Process each query
    results = {
        "metadata": {
            "input_file": input_file,
            "output_file": output_file,
            "processed_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "system_mode": "test_mode" if faq_service.test_mode else "embedding_mode"
        },
        "results": []
    }
    
    matched_count = 0
    
    for i, query in enumerate(queries, 1):
        if not query:
            continue
            
        print(f"🔍 Processing {i}/{len(queries)}: {query[:50]}...")
        
        # Process the query
        result = faq_service.answer_query(query, debug=debug)
        
        # Prepare result for JSON
        query_result = {
            "query_id": i,
            "query": query,
            "found": result.get("found", False),
            "confidence": result.get("confidence", 0.0)
        }
        
        if result.get("found"):
            query_result.update({
                "matched_question": result["matched_question"],
                "answer": result["answer"], 
                "source": result["source"],
                "collection": result.get("collection", "unknown")
            })
            matched_count += 1
            print(f"   ✅ Match found (Confidence: {result['confidence']:.1%})")
        else:
            query_result["message"] = result.get("message", "No match found")
            print(f"   ❌ No match (Best score: {result.get('confidence', 0):.1%})")
        
        # Add debug info if requested
        if debug:
            query_result["debug"] = {
                "detected_collections": result.get("detected_collections", []),
                "candidates": result.get("candidates", [])[:3]  # Top 3 candidates
            }
        
        results["results"].append(query_result)
    
    # Update metadata with summary
    results["metadata"]["matched_count"] = matched_count
    results["metadata"]["match_rate"] = (matched_count / len(queries)) * 100
    
    # Save results to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False
    
    # Print summary
    print("\n" + "="*50)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total queries: {len(queries)}")
    print(f"Matched queries: {matched_count}")
    print(f"Match rate: {results['metadata']['match_rate']:.1f}%")
    print(f"Output file: {output_file}")
    print("="*50)
    
    return True

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Batch process Bengali FAQ queries")
    parser.add_argument("input_file", help="Input text file with queries (one per line)")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    parser.add_argument("-d", "--debug", action="store_true", help="Include debug information")
    parser.add_argument("--stats", action="store_true", help="Show system statistics before processing")
    
    args = parser.parse_args()
    
    print("🚀 Bengali FAQ System - Batch Processor")
    print("=" * 50)
    
    if args.stats:
        stats = faq_service.get_system_stats()
        print("📊 System Statistics:")
        if stats.get('test_mode', False):
            print("   Mode: TEST MODE (no embeddings)")
        else:
            print("   Mode: Full embedding mode")
        print(f"   Collections: {stats.get('total_collections', 0)}")
        total_entries = sum(c['count'] for c in stats.get('collections', {}).values())
        print(f"   Total entries: {total_entries}")
        print()
    
    # Process the batch file
    success = process_batch_file(args.input_file, args.output, args.debug)
    
    if success:
        print("✅ Batch processing completed successfully!")
        sys.exit(0)
    else:
        print("❌ Batch processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 