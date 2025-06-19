#!/usr/bin/env python3
"""
Enhanced Batch Processing System for Bengali FAQ

Features:
- Progress tracking with ETA calculation
- Parallel processing for better performance
- Comprehensive error handling and recovery
- Detailed reporting and analytics
- Memory-efficient processing for large batches
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import threading
from dataclasses import dataclass

from faq_service import faq_service, performance_monitor

@dataclass
class BatchProgress:
    """Progress tracking for batch operations"""
    total: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = 0
    current_query: str = ""
    
    @property
    def success_rate(self) -> float:
        if self.completed == 0:
            return 0.0
        return (self.completed - self.failed) / self.completed * 100
    
    @property
    def eta_seconds(self) -> float:
        if self.completed == 0 or self.total == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed
        remaining = self.total - self.completed
        return remaining / rate if rate > 0 else 0.0
    
    def format_eta(self) -> str:
        eta = self.eta_seconds
        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            return f"{eta/60:.1f}m"
        else:
            return f"{eta/3600:.1f}h"

class ProgressTracker:
    """Thread-safe progress tracking"""
    def __init__(self):
        self.progress = BatchProgress()
        self.lock = threading.Lock()
        self.results = []
        self.errors = []
    
    def start(self, total: int):
        with self.lock:
            self.progress.total = total
            self.progress.start_time = time.time()
    
    def update(self, query: str, success: bool, result: Dict[str, Any]):
        with self.lock:
            self.progress.completed += 1
            self.progress.current_query = query[:50] + "..." if len(query) > 50 else query
            
            if success:
                self.results.append(result)
            else:
                self.progress.failed += 1
                self.errors.append({"query": query, "error": result.get("error", "Unknown error")})
    
    def get_status(self) -> str:
        with self.lock:
            p = self.progress
            return (f"🔄 Progress: {p.completed}/{p.total} "
                   f"({p.completed/p.total*100:.1f}%) - "
                   f"Success: {p.success_rate:.1f}% - "
                   f"ETA: {p.format_eta()} - "
                   f"Current: {p.current_query}")

progress_tracker = ProgressTracker()

def process_single_query(query_data: tuple) -> Dict[str, Any]:
    """Process a single query with error handling"""
    query_id, query, debug = query_data
    
    try:
        start_time = time.time()
        result = faq_service.answer_query(query.strip(), debug=debug)
        duration = time.time() - start_time
        
        # Add metadata
        result.update({
            "query_id": query_id,
            "query": query,
            "processing_time": round(duration, 3)
        })
        
        progress_tracker.update(query, True, result)
        return result
        
    except Exception as e:
        error_result = {
            "query_id": query_id,
            "query": query,
            "found": False,
            "error": f"Processing error: {str(e)}",
            "processing_time": 0.0
        }
        progress_tracker.update(query, False, error_result)
        return error_result

def display_progress():
    """Display live progress updates"""
    while progress_tracker.progress.completed < progress_tracker.progress.total:
        print(f"\r{progress_tracker.get_status()}", end="", flush=True)
        time.sleep(0.5)
    print()  # Final newline

def process_batch_file(input_file: str, output_file: Optional[str] = None, 
                      debug: bool = False, max_workers: int = 4) -> bool:
    """Enhanced batch processing with parallel execution"""
    
    try:
        # Read and validate input file
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file '{input_file}' not found.")
            return False
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            print(f"❌ Error: No queries found in '{input_file}'.")
            return False
        
        print(f"📄 Loaded {len(lines)} queries from '{input_file}'")
        
        # Initialize progress tracking
        progress_tracker.start(len(lines))
        
        # Start progress display in background
        progress_thread = threading.Thread(target=display_progress, daemon=True)
        progress_thread.start()
        
        # Process queries in parallel
        batch_start = time.time()
        query_data = [(i+1, query, debug) for i, query in enumerate(lines)]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_query, data) for data in query_data]
            
            # Wait for all to complete
            for future in as_completed(futures):
                pass  # Results are handled in process_single_query
        
        batch_duration = time.time() - batch_start
        
        # Wait for progress display to finish
        progress_thread.join(timeout=1)
        
        # Generate comprehensive results
        metadata = {
            "input_file": input_file,
            "output_file": output_file,
            "processed_at": datetime.now().isoformat(),
            "total_queries": len(lines),
            "successful_queries": len(progress_tracker.results),
            "failed_queries": progress_tracker.progress.failed,
            "success_rate": progress_tracker.progress.success_rate,
            "total_processing_time": round(batch_duration, 3),
            "average_query_time": round(batch_duration / len(lines), 3),
            "system_mode": "test_mode" if faq_service.test_mode else "embedding_mode",
            "parallel_workers": max_workers,
            "cache_hit_rate": performance_monitor.get_cache_hit_rate(),
            "average_response_time": performance_monitor.get_avg_query_time()
        }
        
        output_data = {
            "metadata": metadata,
            "results": sorted(progress_tracker.results, key=lambda x: x.get("query_id", 0)),
            "errors": progress_tracker.errors if progress_tracker.errors else None
        }
        
        # Remove None values for cleaner output
        output_data = {k: v for k, v in output_data.items() if v is not None}
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        # Display summary
        print("\n" + "="*50)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Total queries: {metadata['total_queries']}")
        print(f"Successful: {metadata['successful_queries']}")
        print(f"Failed: {metadata['failed_queries']}")
        print(f"Success rate: {metadata['success_rate']:.1f}%")
        print(f"Total time: {metadata['total_processing_time']:.2f}s")
        print(f"Average per query: {metadata['average_query_time']:.3f}s")
        print(f"Cache hit rate: {metadata['cache_hit_rate']:.1f}%")
        print(f"Workers used: {metadata['parallel_workers']}")
        
        if progress_tracker.errors:
            print(f"\n⚠️  Errors encountered:")
            for error in progress_tracker.errors[:5]:  # Show first 5 errors
                print(f"  - {error['query'][:50]}: {error['error']}")
            if len(progress_tracker.errors) > 5:
                print(f"  ... and {len(progress_tracker.errors) - 5} more errors")
        
        print("="*50)
        print("✅ Batch processing completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Fatal error during batch processing: {e}")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Batch process Bengali FAQ queries")
    parser.add_argument("input_file", nargs='?', default=None, help="Input text file with queries (one per line). Required unless --stats-only is used.")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    parser.add_argument("-d", "--debug", action="store_true", help="Include debug information")
    parser.add_argument("--stats-only", action="store_true", help="Show system statistics and exit without processing a file.")
    
    args = parser.parse_args()
    
    print("🚀 Bengali FAQ System - Batch Processor")
    print("=" * 50)
    
    # Show stats and exit if --stats-only is used
    if args.stats_only:
        stats = faq_service.get_system_stats()
        print("📊 System Statistics:")
        if stats.get('test_mode', False):
            print("   Mode: TEST MODE (no embeddings)")
        else:
            print("   Mode: Full embedding mode")
        print(f"   Collections: {stats.get('total_collections', 0)}")
        total_entries = sum(c['count'] for c in stats.get('collections', {}).values())
        print(f"   Total entries: {total_entries}")
        print("=" * 50)
        sys.exit(0)
    
    # If not stats-only, input file is required
    if not args.input_file:
        parser.error("the following arguments are required: input_file")

    # Validate that the input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file does not exist: {args.input_file}")
        sys.exit(1)

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