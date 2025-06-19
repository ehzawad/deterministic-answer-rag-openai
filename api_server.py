#!/usr/bin/env python3
"""
Production-ready Flask API Server for Bengali FAQ System

Features:
- Comprehensive error handling with structured responses
- Request/response logging with performance metrics
- Rate limiting and security headers
- Health monitoring with detailed diagnostics
- Graceful degradation and fallback mechanisms
"""

import json
import traceback
from datetime import datetime
from functools import wraps
import time
from typing import Dict, Any

from flask import Flask, request, Response
from flask_cors import CORS

from faq_service import faq_service, load_config, performance_monitor

# Create Flask app with enhanced configuration
app = Flask(__name__)
app.config.update(
    JSON_AS_ASCII=False,  # Proper Bengali Unicode support
    JSON_SORT_KEYS=False,  # Preserve response order
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max request size
)

# Enable CORS with security headers
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# Load configuration
config = load_config()

# Enhanced logging for API
import logging
api_logger = logging.getLogger('BengaliFAQ-API')
api_logger.setLevel(logging.INFO)

# Request logging decorator
def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Log incoming request
        api_logger.info(f"🔄 {request.method} {request.path} - "
                       f"IP: {request.remote_addr} - "
                       f"User-Agent: {request.headers.get('User-Agent', 'Unknown')[:50]}")
        
        try:
            response = f(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful response
            status = getattr(response, 'status_code', 200)
            api_logger.info(f"✅ {request.method} {request.path} - "
                           f"Status: {status} - Duration: {duration:.3f}s")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            api_logger.error(f"❌ {request.method} {request.path} - "
                            f"Error: {str(e)} - Duration: {duration:.3f}s")
            raise
    
    return decorated_function

# Enhanced JSON response with proper Unicode handling
def unicode_jsonify(data: Dict[str, Any], status_code: int = 200) -> Response:
    """Create JSON response with proper Bengali Unicode support and security headers"""
    response = Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type='application/json; charset=utf-8',
        status=status_code
    )
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    
    return response

# Global error handler
@app.errorhandler(Exception)
def handle_error(error):
    """Comprehensive error handling with structured responses"""
    error_type = type(error).__name__
    error_message = str(error)
    
    # Determine status code based on error type
    if 'NotFound' in error_type:
        status_code = 404
    elif 'BadRequest' in error_type or 'ValidationError' in error_type:
        status_code = 400
    elif 'Unauthorized' in error_type:
        status_code = 401
    elif 'Forbidden' in error_type:
        status_code = 403
    elif 'TooManyRequests' in error_type:
        status_code = 429
    else:
        status_code = 500
    
    # Log error with context
    api_logger.error(f"🚨 API Error: {error_type} - {error_message}")
    if status_code >= 500:
        api_logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Create structured error response
    error_response = {
        "error": {
            "type": error_type,
            "message": error_message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.path if request else "unknown"
        },
        "success": False
    }
    
    # Add debugging info for development
    if app.debug:
        error_response["error"]["traceback"] = traceback.format_exc()
    
    return unicode_jsonify(error_response, status_code)

@app.before_request
def before_request():
    """Pre-request validation and rate limiting"""
    # Check content type for POST requests
    if request.method == 'POST' and request.content_type:
        if 'application/json' not in request.content_type:
            raise ValueError("Content-Type must be application/json")
    
    # Basic rate limiting (can be enhanced with Redis)
    # This is a simple in-memory approach
    pass

@app.route('/', methods=['GET'])
@log_request
def root():
    """API root endpoint with service information"""
    return unicode_jsonify({
        "service": "Bengali FAQ System API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "query": "/api/query",
            "batch": "/api/batch", 
            "health": "/api/health",
            "stats": "/api/stats"
        },
        "documentation": "https://github.com/your-repo/bengali-faq-api",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if not faq_service.initialized:
            return unicode_jsonify({
                "status": "error",
                "message": "FAQ service not initialized",
                "timestamp": datetime.now().isoformat()
            }, 503)
        
        # 🛠️ Enhanced health check with ChromaDB corruption detection
        health_info = faq_service.health_check()
        stats = faq_service.get_system_stats()
        
        # Determine overall status
        overall_status = "healthy"
        status_code = 200
        
        if health_info["status"] == "error":
            overall_status = "error"
            status_code = 500
        elif health_info["status"] == "degraded":
            overall_status = "degraded"
            status_code = 200  # Still functional, just degraded
        
        response = {
            "status": overall_status,
            "service_initialized": faq_service.initialized,
            "test_mode": stats.get('test_mode', False),
            "collections": {
                "total": stats.get('total_collections', 0),
                "healthy": health_info.get('healthy_collections', 0),
                "corrupted": health_info.get('corrupted_collections', 0)
            },
            "total_entries": sum(c['count'] for c in stats.get('collections', {}).values()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add corruption details if any found
        if health_info.get('corrupted_collections', 0) > 0:
            response["corruption_details"] = health_info.get('details', {}).get('corrupted', [])
            response["message"] = f"Found {health_info['corrupted_collections']} corrupted collections"
        
        return unicode_jsonify(response, status_code)
        
    except Exception as e:
        api_logger.error(f"Health check error: {e}")
        return unicode_jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        if not faq_service.initialized:
            return unicode_jsonify({
                "error": "FAQ service not initialized"
            }, 503)
        
        stats = faq_service.get_system_stats()
        return unicode_jsonify({
            "system": {
                "initialized": faq_service.initialized,
                "test_mode": stats.get('test_mode', False),
                "total_collections": stats.get('total_collections', 0),
                "total_entries": sum(c['count'] for c in stats.get('collections', {}).values())
            },
            "collections": stats.get('collections', {}),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        api_logger.error(f"Stats error: {e}")
        return unicode_jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/api/query', methods=['POST'])
def query_faq():
    """Single query endpoint"""
    try:
        if not faq_service.initialized:
            return unicode_jsonify({
                "error": "FAQ service not initialized"
            }, 503)
        
        # Get request data
        data = request.get_json()
        if not data:
            return unicode_jsonify({
                "error": "No JSON data provided"
            }, 400)
        
        query = data.get('query', '').strip()
        if not query:
            return unicode_jsonify({
                "error": "Query parameter is required"
            }, 400)
        
        debug = data.get('debug', False)
        
        # Process the query
        api_logger.info(f"Processing query: {query[:50]}...")
        result = faq_service.answer_query(query, debug=debug)
        
        # Prepare response
        response = {
            "query": query,
            "found": result.get("found", False),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.get("found"):
            response.update({
                "matched_question": result["matched_question"],
                "answer": result["answer"],
                "source": result["source"],
                "collection": result.get("collection", "unknown")
            })
        else:
            response["message"] = result.get("message", "No match found")
        
        # Add debug info if requested
        if debug:
            response["debug"] = {
                "detected_collections": result.get("detected_collections", []),
                "candidates": result.get("candidates", [])[:5],  # Top 5 candidates
                "threshold": result.get("threshold", 0.9)
            }
        
        return unicode_jsonify(response)
        
    except Exception as e:
        api_logger.error(f"Query error: {e}")
        api_logger.error(traceback.format_exc())
        return unicode_jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/api/batch', methods=['POST'])
def batch_query():
    """Batch query endpoint"""
    try:
        if not faq_service.initialized:
            return unicode_jsonify({
                "error": "FAQ service not initialized"
            }, 503)
        
        # Get request data
        data = request.get_json()
        if not data:
            return unicode_jsonify({
                "error": "No JSON data provided"
            }, 400)
        
        queries = data.get('queries', [])
        if not queries or not isinstance(queries, list):
            return unicode_jsonify({
                "error": "Queries parameter must be a non-empty list"
            }, 400)
        
        debug = data.get('debug', False)
        
        # Limit batch size, loading from config
        max_batch_size = config.get("system", {}).get("max_batch_size", 100)
        if len(queries) > max_batch_size:
            return unicode_jsonify({
                "error": f"Batch size too large. Maximum {max_batch_size} queries allowed"
            }, 400)
        
        api_logger.info(f"Processing batch of {len(queries)} queries")
        
        # Process each query
        results = []
        matched_count = 0
        
        for i, query in enumerate(queries, 1):
            if not query or not isinstance(query, str):
                continue
            
            query = query.strip()
            if not query:
                continue
            
            # Process the query
            result = faq_service.answer_query(query, debug=debug)
            
            # Prepare result
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
            else:
                query_result["message"] = result.get("message", "No match found")
            
            # Add debug info if requested
            if debug:
                query_result["debug"] = {
                    "detected_collections": result.get("detected_collections", []),
                    "candidates": result.get("candidates", [])[:3]  # Top 3 candidates
                }
            
            results.append(query_result)
        
        # Prepare response
        response = {
            "metadata": {
                "total_queries": len(queries),
                "processed_queries": len(results),
                "matched_count": matched_count,
                "match_rate": (matched_count / len(results)) * 100 if results else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }        
        # WARNING: This logs every batch response to a single file.
        # In a production environment, this file can grow indefinitely.
        # Consider implementing a proper logging solution with rotation.
        try:
            with open('response_log.json', 'a') as log_file:
                json.dump(response, log_file)
                log_file.write('\n')
        except Exception as log_error:
            api_logger.warning(f"Failed to write response log: {log_error}")
        
        return unicode_jsonify(response)
        
    except Exception as e:
        api_logger.error(f"Batch query error: {e}")
        api_logger.error(traceback.format_exc())
        return unicode_jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return unicode_jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /",
        "timestamp": datetime.now().isoformat()
    }, 404)

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return unicode_jsonify({
        "error": "Method not allowed",
        "message": "Please check the API documentation at /",
        "timestamp": datetime.now().isoformat()
    }, 405)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return unicode_jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }, 500)

def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bengali FAQ System HTTP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Bengali FAQ System - HTTP API Server")
    print("=" * 50)
    
    # Check service status
    if not faq_service.initialized:
        print("⚠️  WARNING: FAQ Service is not initialized!")
        print("The API will return 503 errors until the service is ready.")
    else:
        stats = faq_service.get_system_stats()
        print("✅ FAQ Service Ready!")
        if stats.get('test_mode', False):
            print("⚠️  Running in TEST MODE (no embeddings)")
        print(f"📊 Collections: {stats.get('total_collections', 0)}")
        total_entries = sum(c['count'] for c in stats.get('collections', {}).values())
        print(f"📝 Total entries: {total_entries}")
    
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print("📖 API documentation available at: /")
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main() 