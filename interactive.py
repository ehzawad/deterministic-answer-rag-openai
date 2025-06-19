#!/usr/bin/env python3
"""
Enhanced Interactive CLI for Bengali FAQ System

Features:
- Command history and auto-completion
- Rich formatting and colored output
- Built-in help system and examples
- Performance monitoring and statistics
- Session management and query history
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import atexit

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

from faq_service import faq_service, performance_monitor

class InteractiveSession:
    """Enhanced interactive session with history and statistics"""
    
    def __init__(self):
        self.query_history = []
        self.session_start = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        
        # Setup readline for command history
        if HAS_READLINE:
            self.setup_readline()
    
    def setup_readline(self):
        """Configure readline for better UX"""
        # History file
        histfile = os.path.join(os.path.expanduser("~"), ".bengali_faq_history")
        
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        # Save history on exit
        atexit.register(readline.write_history_file, histfile)
        
        # Tab completion (basic)
        readline.parse_and_bind('tab: complete')
        
        # Enable history search
        readline.parse_and_bind('"\e[A": history-search-backward')
        readline.parse_and_bind('"\e[B": history-search-forward')
    
    def add_query(self, query: str, result: Dict, duration: float):
        """Record a query in session history"""
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "found": result.get("found", False),
            "confidence": result.get("confidence", 0.0),
            "duration": duration
        })
        
        self.total_queries += 1
        if result.get("found", False):
            self.successful_queries += 1
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        session_duration = time.time() - self.session_start
        
        return {
            "session_duration": f"{session_duration/60:.1f} minutes",
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": f"{(self.successful_queries/self.total_queries*100):.1f}%" if self.total_queries > 0 else "0%",
            "avg_response_time": f"{performance_monitor.get_avg_query_time():.3f}s",
            "cache_hit_rate": f"{performance_monitor.get_cache_hit_rate():.1f}%"
        }

def print_banner():
    """Display enhanced welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              🇧🇩 Bengali FAQ System - Interactive Mode         ║
║                   Deterministic Answer Engine                 ║
╠══════════════════════════════════════════════════════════════╣
║ Commands:                                                    ║
║   📝 Type your question in Bengali                           ║
║   📊 'stats'     - Show session statistics                   ║
║   📈 'system'    - Show system information                   ║
║   📋 'history'   - Show query history                        ║
║   🔧 'debug on'  - Enable debug mode                         ║
║   🔇 'debug off' - Disable debug mode                        ║
║   💾 'export'    - Export session history                    ║
║   ❓ 'help'      - Show detailed help                         ║
║   🚪 'exit'      - Exit the application                      ║
║                                                              ║
║ Tips:                                                        ║
║   • Press ↑/↓ for command history                           ║
║   • Use Ctrl+C to interrupt at any time                     ║
║   • Type 'help examples' for sample queries                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_help(topic: Optional[str] = None):
    """Enhanced help system"""
    if topic == "examples":
        print("""
📖 Sample Bengali Queries:

Banking & Accounts:
• পেরোল একাউন্টের সুবিধাসমূহ কী কী?
• ইয়াকিন সেভিংস একাউন্টে সর্বনিম্ন কত টাকা লাগে?
• এটিএম কার্ডের চার্জ কত?

Cards & Services:
• ক্রেডিট কার্ড পেতে কী কী কাগজপত্র লাগে?
• ডেবিট কার্ডের বার্ষিক ফি কত?
• মোবাইল ব্যাংকিং সেবা কীভাবে ব্যবহার করব?

General:
• শাখার ঠিকানা জানতে চাই
• গ্রাহক সেবা নম্বর কত?
• অনলাইনে একাউন্ট খোলা যায় কি?
""")
    elif topic == "debug":
        print("""
🔧 Debug Mode Information:

When debug mode is enabled, you'll see:
• Detected collections for routing
• Top candidate matches with scores
• Semantic boost applications
• Performance metrics
• Threshold calculations

This is useful for understanding how the system
processes your queries and makes decisions.
""")
    else:
        print("""
❓ Detailed Help:

🎯 Query Processing:
• The system uses advanced hybrid matching
• Combines semantic understanding with exact matches
• Provides confidence scores for all responses
• Supports Bengali text normalization

📊 Statistics Commands:
• 'stats' - Current session performance
• 'system' - FAQ database information
• 'history' - Your recent queries

🔧 Debug Features:
• 'debug on/off' - Toggle detailed processing info
• Shows collection routing and scoring details

💾 Export Options:
• 'export' - Save current session as JSON
• Includes all queries, responses, and timing

🚀 Performance:
• Built-in caching for faster responses
• Real-time performance monitoring
• Session tracking and analytics

Type 'help examples' for sample queries
Type 'help debug' for debug mode details
""")

def format_result(result: Dict, debug: bool = False) -> str:
    """Format query result with rich formatting"""
    if result.get("found"):
        confidence = result.get("confidence", 0.0)
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🟠"
        
        output = f"""
{confidence_emoji} Match Found (Confidence: {confidence:.1%})

📝 Question: {result.get('matched_question', 'Unknown')}
📋 Answer: {result.get('answer', 'No answer available')}
📁 Source: {result.get('source', 'Unknown')}
📊 Collection: {result.get('collection', 'Unknown')}
"""
        
        if debug and result.get("debug"):
            debug_info = result["debug"]
            output += f"""
🔧 Debug Information:
   • Detected Collections: {debug_info.get('detected_collections', [])}
   • Processing Time: {debug_info.get('processing_time', 0):.3f}s
   • Top Candidates: {len(debug_info.get('candidates', []))}
"""
    else:
        output = f"""
❌ No Match Found

📝 Query: {result.get('query', 'Unknown')}
💭 Message: {result.get('message', 'No suitable answer found in the knowledge base.')}
📊 Best Score: {result.get('confidence', 0.0):.1%}

💡 Suggestions:
• Try rephrasing your question
• Use simpler Bengali terms
• Check for spelling errors
• Type 'help examples' for sample queries
"""
    
    return output

def handle_command(command: str, session: InteractiveSession, debug: bool) -> tuple:
    """Handle special commands"""
    command = command.lower().strip()
    
    if command == "stats":
        stats = session.get_session_stats()
        print(f"""
📊 Session Statistics:
   • Duration: {stats['session_duration']}
   • Total Queries: {stats['total_queries']}
   • Successful: {stats['successful_queries']}
   • Success Rate: {stats['success_rate']}
   • Avg Response: {stats['avg_response_time']}
   • Cache Hit Rate: {stats['cache_hit_rate']}
""")
        return debug, True
    
    elif command == "system":
        system_stats = faq_service.get_system_stats()
        print(f"""
🖥️  System Information:
   • Initialized: {faq_service.initialized}
   • Test Mode: {faq_service.test_mode}
   • Total Collections: {system_stats.get('total_collections', 0)}
   • Total Entries: {sum(c['count'] for c in system_stats.get('collections', {}).values())}
   • Mode: {system_stats.get('mode', 'Unknown')}
""")
        return debug, True
    
    elif command == "history":
        if not session.query_history:
            print("📝 No queries in current session")
        else:
            print(f"📋 Query History ({len(session.query_history)} queries):")
            for i, query in enumerate(session.query_history[-10:], 1):  # Last 10
                status = "✅" if query["found"] else "❌"
                print(f"   {i}. {status} {query['query'][:50]}... ({query['confidence']:.1%})")
        return debug, True
    
    elif command.startswith("debug"):
        if "on" in command:
            print("🔧 Debug mode enabled")
            return True, True
        elif "off" in command:
            print("🔇 Debug mode disabled")
            return False, True
        else:
            print(f"🔧 Debug mode is currently {'ON' if debug else 'OFF'}")
            return debug, True
    
    elif command == "export":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_export_{timestamp}.json"
        
        export_data = {
            "session_info": session.get_session_stats(),
            "query_history": session.query_history,
            "exported_at": datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Session exported to: {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
        
        return debug, True
    
    elif command.startswith("help"):
        topic = command.split(maxsplit=1)[1] if len(command.split()) > 1 else None
        print_help(topic)
        return debug, True
    
    elif command in ["exit", "quit", "q"]:
        return debug, False
    
    else:
        print(f"❓ Unknown command: {command}")
        print("Type 'help' for available commands")
        return debug, True

def main():
    """Enhanced main interactive loop"""
    # Initialize session
    session = InteractiveSession()
    debug_mode = False
    
    print_banner()
    
    # Check system status
    if not faq_service.initialized:
        print("🔄 Initializing FAQ service...")
        if not faq_service.initialize():
            print("❌ Failed to initialize FAQ service")
            return
        print("✅ FAQ service initialized successfully")
    
    # Display system info
    mode_info = "🧪 Test Mode" if faq_service.test_mode else "🚀 Production Mode"
    print(f"\n{mode_info} | Ready to answer your questions!")
    print("="*50)
    
    # Main interaction loop
    while True:
        try:
            # Create prompt with current status
            debug_indicator = " (debug)" if debug_mode else ""
            prompt = f"🔍 Enter your query{debug_indicator} » "
            
            # Get user input
            try:
                user_input = input(prompt).strip()
            except EOFError:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if it's a command
            if user_input.lower().startswith(('stats', 'system', 'history', 'debug', 'export', 'help', 'exit', 'quit')):
                debug_mode, should_continue = handle_command(user_input, session, debug_mode)
                if not should_continue:
                    break
                continue
            
            # Process as FAQ query
            print("🔄 Processing your query...")
            start_time = time.time()
            
            result = faq_service.answer_query(user_input, debug=debug_mode)
            
            duration = time.time() - start_time
            session.add_query(user_input, result, duration)
            
            # Display result
            print(format_result(result, debug_mode))
            
            # Record performance
            performance_monitor.record_query_time(duration)
            
            print("="*50)
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Interrupted. Type 'exit' to quit or continue with another query.")
            continue
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'exit' to quit.")
    
    # Session summary
    stats = session.get_session_stats()
    print(f"""
📊 Session Summary:
   • Duration: {stats['session_duration']}
   • Queries Processed: {stats['total_queries']}
   • Success Rate: {stats['success_rate']}

Thank you for using the Bengali FAQ System! 🙏
""")

if __name__ == "__main__":
    main() 