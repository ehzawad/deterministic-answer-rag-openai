import os
import json
import pickle
import hashlib
import asyncio
import logging
import traceback
import glob
from typing import List, Dict, Optional, Set, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BengaliFAQ-RAG')

# Load environment variables from .env file if present
load_dotenv(verbose=False)

# Constants
FAQ_DIR = "faq_data"  # Updated path to match user's directory
CACHE_DIR = "cache"  # Directory for caching
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
CONTENT_CACHE_FILE = os.path.join(CACHE_DIR, "content_cache.pkl")
FILE_HASH_CACHE = os.path.join(CACHE_DIR, "file_hashes.json")
MAX_CANDIDATES = 5  # Maximum number of candidates to consider for final answer
MODEL = "gpt-4.1-mini"     # Updated to use gpt-4.1-mini for answering questions


class OptimizedBengaliFAQSystem:
    """Optimized Bengali FAQ System using pure semantic matching"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.faq_data = []  # Stores all FAQ entries in memory
        self.embeddings = []  # Stores pre-computed embeddings
        self.file_hashes = {}  # Tracks file changes
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Load cached data if available
        self._load_cache()
    
    def _discover_faq_files(self) -> List[str]:
        """Dynamically discover all .txt files in the FAQ directory"""
        try:
            if not os.path.exists(FAQ_DIR):
                logger.warning(f"FAQ directory '{FAQ_DIR}' does not exist.")
                return []
            
            # Find all .txt files in the FAQ directory
            txt_files = glob.glob(os.path.join(FAQ_DIR, "*.txt"))
            
            # Extract just the filenames (not full paths) for consistency
            filenames = [os.path.basename(filepath) for filepath in txt_files]
            
            logger.info(f"Discovered {len(filenames)} .txt files in {FAQ_DIR}: {filenames}")
            return filenames
            
        except Exception as e:
            logger.error(f"Error discovering FAQ files: {e}")
            return []
    
    def _load_cache(self):
        """Load cached FAQ data and embeddings"""
        # Load file hashes
        if os.path.exists(FILE_HASH_CACHE):
            try:
                with open(FILE_HASH_CACHE, 'r') as f:
                    self.file_hashes = json.load(f)
                logger.info(f"Loaded {len(self.file_hashes)} cached file hashes")
            except Exception as e:
                logger.error(f"Error loading cached file hashes: {e}")
                self.file_hashes = {}
        
        # Load content cache
        if os.path.exists(CONTENT_CACHE_FILE):
            try:
                with open(CONTENT_CACHE_FILE, 'rb') as f:
                    self.faq_data = pickle.load(f)
                logger.info(f"Loaded {len(self.faq_data)} FAQ entries from cache")
            except Exception as e:
                logger.error(f"Error loading content cache: {e}")
                self.faq_data = []
        
        # Load embedding cache
        if os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {e}")
                self.embeddings = []
    
    def _save_cache(self):
        """Save FAQ data and embeddings to cache"""
        # Save file hashes
        try:
            with open(FILE_HASH_CACHE, 'w') as f:
                json.dump(self.file_hashes, f)
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")
        
        # Save content cache
        try:
            with open(CONTENT_CACHE_FILE, 'wb') as f:
                pickle.dump(self.faq_data, f)
        except Exception as e:
            logger.error(f"Error saving content cache: {e}")
        
        # Save embedding cache
        try:
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
        
        logger.info(f"Cache saved successfully: {len(self.faq_data)} FAQ entries, {len(self.embeddings)} embeddings")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
                return hashlib.md5(file_data).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def _check_for_updates(self) -> Tuple[bool, Set[str]]:
        """Check if FAQ files have been modified since last run
        
        Returns:
            Tuple[bool, Set[str]]: (any_updates, files_to_process)
        """
        files_to_process = set()
        
        # Get all discovered FAQ files
        discovered_files = self._discover_faq_files()
        
        for filename in discovered_files:
            filepath = os.path.join(FAQ_DIR, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Warning: File {filepath} does not exist.")
                continue
                
            current_hash = self._calculate_file_hash(filepath)
            if current_hash:
                # Check if file is new or modified
                if filename not in self.file_hashes or self.file_hashes[filename] != current_hash:
                    files_to_process.add(filename)
                    self.file_hashes[filename] = current_hash
        
        return len(files_to_process) > 0, files_to_process
    
    def _preprocess_faq_file(self, filepath: str) -> List[Dict[str, str]]:
        """Preprocess FAQ file to extract Q&A pairs with special handling for Bengali text"""
        try:
            # Ensure proper UTF-8 encoding for Bengali text
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Bengali or English question/answer markers
            q_markers = ["Question:", "প্রশ্ন:"]  # English and Bengali
            a_markers = ["Answer:", "উত্তর:"]     # English and Bengali
            
            # Split content into Q&A pairs
            pairs = []
            current_question = None
            current_answer = []
            in_answer = False
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Check for question markers (both English and Bengali)
                is_question_line = any(line.startswith(marker) for marker in q_markers)
                is_answer_line = any(line.startswith(marker) for marker in a_markers)
                
                if is_question_line:
                    # Save previous Q&A pair if exists
                    if current_question and current_answer:
                        pairs.append({
                            "question": current_question,
                            "answer": ' '.join(current_answer),
                            "source": os.path.basename(filepath)
                        })
                    
                    # Extract the question text after the marker
                    for marker in q_markers:
                        if line.startswith(marker):
                            current_question = line[len(marker):].strip()
                            break
                    
                    current_answer = []
                    in_answer = False
                
                elif is_answer_line:
                    # Extract the answer text after the marker
                    for marker in a_markers:
                        if line.startswith(marker):
                            answer_text = line[len(marker):].strip()
                            if answer_text:
                                current_answer.append(answer_text)
                            break
                    
                    in_answer = True
                
                elif in_answer and line:
                    current_answer.append(line)
                
                elif not in_answer and line:
                    if current_question:
                        current_question += " " + line
                    else:
                        current_question = line
            
            # Add the last pair if it exists
            if current_question and current_answer:
                pairs.append({
                    "question": current_question,
                    "answer": ' '.join(current_answer),
                    "source": os.path.basename(filepath)
                })
            
            # Debug output to verify extraction
            logger.info(f"Extracted {len(pairs)} Q&A pairs from {filepath}")
            
            return pairs
            
        except UnicodeDecodeError as ue:
            logger.error(f"Unicode error processing {filepath} - possible Bengali encoding issue: {ue}")
            # Try with a different encoding as fallback
            try:
                with open(filepath, 'r', encoding='utf-16') as f:
                    content = f.read()
                logger.info(f"Successfully read file with utf-16 encoding")
                # Process the content directly
                return self._process_content(content, filepath)
            except Exception as e2:
                logger.error(f"Failed with alternative encoding too: {e2}")
                return []
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return []
    
    def _process_content(self, content: str, filepath: str) -> List[Dict[str, str]]:
        """Helper method to process file content"""
        # Similar logic as in _preprocess_faq_file but works with content string directly
        q_markers = ["Question:", "প্রশ্ন:"]  # English and Bengali
        a_markers = ["Answer:", "উত্তর:"]     # English and Bengali
        
        pairs = []
        current_question = None
        current_answer = []
        in_answer = False
        
        for line in content.split('\n'):
            line = line.strip()
            is_question_line = any(line.startswith(marker) for marker in q_markers)
            is_answer_line = any(line.startswith(marker) for marker in a_markers)
            
            if is_question_line:
                if current_question and current_answer:
                    pairs.append({
                        "question": current_question,
                        "answer": ' '.join(current_answer),
                        "source": os.path.basename(filepath)
                    })
                
                for marker in q_markers:
                    if line.startswith(marker):
                        current_question = line[len(marker):].strip()
                        break
                
                current_answer = []
                in_answer = False
            
            elif is_answer_line:
                for marker in a_markers:
                    if line.startswith(marker):
                        answer_text = line[len(marker):].strip()
                        if answer_text:
                            current_answer.append(answer_text)
                        break
                
                in_answer = True
            
            elif in_answer and line:
                current_answer.append(line)
            
            elif not in_answer and line:
                if current_question:
                    current_question += " " + line
                else:
                    current_question = line
        
        if current_question and current_answer:
            pairs.append({
                "question": current_question,
                "answer": ' '.join(current_answer),
                "source": os.path.basename(filepath)
            })
        
        logger.info(f"Extracted {len(pairs)} Q&A pairs from content")
        return pairs
    
    def _create_embeddings_for_questions(self, questions: List[str]) -> List[List[float]]:
        """Create embeddings for a list of questions in a single batch request"""
        if not questions:
            logger.warning("No questions to embed")
            return []
            
        try:
            # Use text-embedding-3-large for better multilingual performance
            logger.info(f"Creating embeddings for {len(questions)} questions using text-embedding-3-large")
            
            # Process in batches if there are many questions
            MAX_BATCH_SIZE = 100  # Maximum number of texts to embed in one request
            all_embeddings = []
            
            for i in range(0, len(questions), MAX_BATCH_SIZE):
                batch = questions[i:i+MAX_BATCH_SIZE]
                logger.info(f"Processing batch {i//MAX_BATCH_SIZE + 1} with {len(batch)} questions")
                
                response = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch,
                    dimensions=1024  # Reduced dimensions for efficiency while maintaining quality
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//MAX_BATCH_SIZE + 1} - received {len(batch_embeddings)} embeddings")
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _embed_single_query(self, query: str) -> List[float]:
        """Create embedding for a single query"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=[query],
                dimensions=1024  # Reduced dimensions for efficiency
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not a or not b:
            return 0
            
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def _find_candidate_matches(self, query_embedding: List[float], top_k: int = MAX_CANDIDATES) -> List[Dict]:
        """Find top candidate matches using embedding similarity"""
        if not query_embedding or not self.embeddings or len(self.embeddings) != len(self.faq_data):
            logger.warning("Cannot find candidates: missing embeddings or FAQ data")
            return []
        
        # Calculate similarities
        similarities = [
            (i, self._cosine_similarity(query_embedding, embedding))
            for i, embedding in enumerate(self.embeddings)
        ]
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top candidates
        candidates = []
        for i, score in similarities[:top_k]:
            candidates.append({
                "question": self.faq_data[i]["question"],
                "answer": self.faq_data[i]["answer"],
                "source": self.faq_data[i]["source"],
                "score": score
            })
        
        return candidates
    
    def initialize(self) -> bool:
        """Initialize the system by loading and preprocessing all FAQ data"""
        try:
            logger.info("Initializing Optimized Bengali FAQ System...")
            
            # Check if FAQ directory exists
            if not os.path.exists(FAQ_DIR):
                logger.error(f"Error: FAQ directory '{FAQ_DIR}' does not exist.")
                logger.info(f"Please ensure your FAQ files are in: {FAQ_DIR}")
                return False
            
            # Discover all FAQ files dynamically
            discovered_files = self._discover_faq_files()
            if not discovered_files:
                logger.error("No .txt files found in FAQ directory. Please add FAQ files.")
                return False
            
            # Log discovered files
            logger.info(f"Found {len(discovered_files)} FAQ files: {discovered_files}")
            
            # Check if we need to process any files
            updates_needed, files_to_process = self._check_for_updates()
            
            # If no updates needed and we have cached data, we're good to go
            if not updates_needed and self.faq_data and self.embeddings and len(self.faq_data) == len(self.embeddings):
                logger.info("No updates needed. Using cached data.")
                return True
            
            # If no files to process or if we need to force processing all files
            if not files_to_process or not self.faq_data:
                logger.info("Processing all discovered files...")
                files_to_process = set(discovered_files)
            
            # Process files
            logger.info(f"Processing {len(files_to_process)} files...")
            
            # We'll store the processed FAQ entries
            new_faq_entries = []
            
            # Process each file that needs updating
            for filename in files_to_process:
                filepath = os.path.join(FAQ_DIR, filename)
                if os.path.exists(filepath):
                    logger.info(f"Processing file: {filepath}")
                    faq_pairs = self._preprocess_faq_file(filepath)
                    if faq_pairs:
                        logger.info(f"Extracted {len(faq_pairs)} FAQ pairs from {filepath}")
                        new_faq_entries.extend(faq_pairs)
                    else:
                        logger.warning(f"No FAQ pairs extracted from {filepath}")
            
            # If we have cached entries, remove entries from files we're updating and add new entries
            if self.faq_data and files_to_process != set(discovered_files):
                # Remove old entries from files we're updating
                self.faq_data = [
                    entry for entry in self.faq_data 
                    if entry["source"] not in files_to_process
                ]
                
                # Add new entries
                self.faq_data.extend(new_faq_entries)
            else:
                # No cached entries, just use the new ones
                self.faq_data = new_faq_entries
            
            logger.info(f"Total FAQ entries: {len(self.faq_data)}")
            
            if not self.faq_data:
                logger.error("No FAQ data extracted from files. Check file format.")
                return False
            
            # Now we need to update embeddings
            
            # Get all questions
            questions = [entry["question"] for entry in self.faq_data]
            
            # Create embeddings for all questions in a single batch request
            logger.info(f"Creating embeddings for {len(questions)} questions...")
            self.embeddings = self._create_embeddings_for_questions(questions)
            
            if not self.embeddings or len(self.embeddings) != len(questions):
                logger.error(f"Failed to create embeddings. Got {len(self.embeddings)} embeddings for {len(questions)} questions.")
                return False
            
            # Save cache
            logger.info("Saving updated cache...")
            self._save_cache()
            
            logger.info(f"Initialization complete. {len(self.faq_data)} FAQ entries loaded and embedded.")
            return len(self.faq_data) > 0
            
        except Exception as e:
            logger.error(f"Error initializing Optimized Bengali FAQ System: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def answer_query_async(self, query: str, debug: bool = False) -> Dict:
        """Answer a user query by finding the most semantically similar question in the FAQ database"""
        try:
            if not self.faq_data or not self.embeddings:
                return {"found": False, "message": "FAQ system not initialized."}
            
            # Check if query is in Bengali
            has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
            
            # Create embedding directly for the original query - no preprocessing
            logger.info("Creating embedding for query...")
            query_embedding = self._embed_single_query(query)
            
            if not query_embedding:
                return {"found": False, "message": "Failed to create query embedding."}
            
            # Find candidate matches using semantic search (question-to-question similarity)
            logger.info("Finding semantic matches...")
            candidates = self._find_candidate_matches(query_embedding, top_k=5)  # Get top 5 candidates
            
            if not candidates:
                # Return message in Bengali if query was in Bengali
                if has_bengali:
                    return {
                        "found": False,
                        "message": "কোন মিল পাওয়া যায়নি। অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লিখুন।"  # No match found. Please rephrase your question.
                    }
                else:
                    return {
                        "found": False,
                        "message": "No matching FAQ entries found. Please rephrase your question."
                    }
            
            # Now use GPT-4o-mini to determine which candidate question is semantically most similar
            logger.info("Using GPT-4o-mini for question-to-question semantic matching...")
            
            # Create a formatted list of candidate questions
            candidates_text = ""
            for i, candidate in enumerate(candidates):
                candidates_text += f"Question {i+1}: {candidate['question']}\n"
            
            # Create a prompt focused purely on question-to-question semantic matching
            selection_prompt = f"""Determine which of these FAQ questions is most semantically similar to the user's question.

User Question: {query}

Candidate FAQ Questions:
{candidates_text}

Focus only on the semantic meaning and intent of the questions. Ignore superficial differences in phrasing or language structure. The goal is to find which question is asking for the same information, even if phrased differently.

Return your analysis in this format:

MATCH: [just the number of the best question match, e.g., 2]
CONFIDENCE: [number between 0.0-1.0]
REASONING: [brief explanation of why you chose this question as the best match]"""

            # Try using the new Responses API first, fallback to Chat Completions if needed
            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": selection_prompt
                                }
                            ]
                        }
                    ],
                    temperature=0
                )
                result_text = response.output_text
                logger.info("Successfully used Responses API")
                
            except Exception as responses_error:
                logger.warning(f"Responses API failed: {responses_error}")
                logger.info("Falling back to Chat Completions API...")
                
                # Fallback to Chat Completions API
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": selection_prompt}
                    ],
                    temperature=0
                )
                result_text = response.choices[0].message.content
                logger.info("Successfully used Chat Completions API fallback")
            
            # Parse the result with improved robustness
            match_num = None
            confidence = 0.0
            reasoning = ""
            
            # Extract match number with better error handling
            if "MATCH:" in result_text:
                match_section = result_text.split("MATCH:")[1].split("\n")[0].strip()
                # Handle various formats: "2", "Question 2", "#2", etc.
                import re
                match_pattern = r"(?:Question\s*)?(\d+)"
                match = re.search(match_pattern, match_section)
                if match:
                    match_num = int(match.group(1))
                    logger.info(f"Extracted match number: {match_num}")
                elif "NO_MATCH" in match_section.upper():
                    match_num = None
                    logger.info("No match found")
            
            # Extract confidence
            if "CONFIDENCE:" in result_text:
                confidence_section = result_text.split("CONFIDENCE:")[1].split("\n")[0].strip()
                # Extract any number between 0 and 1
                import re
                confidence_pattern = r"(0\.\d+|\d\.\d+|1\.0|1|0)"
                match = re.search(confidence_pattern, confidence_section)
                if match:
                    confidence = float(match.group(1))
                    logger.info(f"Extracted confidence: {confidence}")
            
            # Extract reasoning
            if "REASONING:" in result_text:
                reasoning_section = result_text.split("REASONING:")[1].strip()
                reasoning = reasoning_section
            
            # Determine if a match was found
            if match_num is not None and 0 <= match_num - 1 < len(candidates):
                found = True
                matched_question = candidates[match_num - 1]["question"]
                # Use the corresponding answer directly from the FAQ
                matched_answer = candidates[match_num - 1]["answer"]
                source = candidates[match_num - 1]["source"]
            else:
                found = False
                matched_question = ""
                matched_answer = ""
                source = ""
            
            # Prepare the result
            result = {
                "found": found,
                "confidence": confidence
            }
            
            if found:
                result["matched_question"] = matched_question
                result["answer"] = matched_answer
                result["source"] = source
                if reasoning:
                    result["reasoning"] = reasoning
            else:
                # Return message in Bengali if query was in Bengali
                if has_bengali:
                    result["message"] = "কোন মিল পাওয়া যায়নি। অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লিখুন।"
                else:
                    result["message"] = "No matching FAQ entries found. Please rephrase your question."
            
            # Add debug info if requested
            if debug:
                result["candidates"] = candidates
                result["gpt_response"] = result_text
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            logger.error(traceback.format_exc())
            return {"found": False, "message": f"Error processing query: {str(e)}"}
    
    def answer_query(self, query: str, debug: bool = False) -> Dict:
        """Synchronous wrapper for answer_query_async"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.answer_query_async(query, debug)
            )
        finally:
            loop.close()
    
    def clear_cache(self) -> bool:
        """Clear all cached data"""
        try:
            # Clear in-memory data
            self.faq_data = []
            self.embeddings = []
            self.file_hashes = {}
            
            # Delete cache files
            for cache_file in [EMBEDDING_CACHE_FILE, CONTENT_CACHE_FILE, FILE_HASH_CACHE]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


async def main_async():
    # Initialize the FAQ system
    faq_system = OptimizedBengaliFAQSystem()
    initialization_success = faq_system.initialize()
    
    try:
        # Interactive query testing
        print("\nOptimized Bengali FAQ System")
        print("Type 'exit' to quit")
        print("Type 'debug on' to enable debug mode")
        print("Type 'debug off' to disable debug mode")
        print("Type 'clear cache' to clear all cached data")
        
        if not initialization_success:
            print("\nWARNING: FAQ system was not initialized properly.")
            print("Queries will likely fail. Please check error messages above.")
            print("Make sure your FAQ files are in the correct directory.")
            print("You can still try queries, but they may not work correctly.")
        
        debug_mode = False
        exit_requested = False
        
        while not exit_requested:
            try:
                query = input("\nEnter your query (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    exit_requested = True
                    print("\nExiting FAQ system...")
                    continue
                elif query.lower() == 'debug on':
                    debug_mode = True
                    print("Debug mode enabled.")
                    continue
                elif query.lower() == 'debug off':
                    debug_mode = False
                    print("Debug mode disabled.")
                    continue
                elif query.lower() == 'clear cache':
                    faq_system.clear_cache()
                    continue
                
                result = await faq_system.answer_query_async(query, debug=debug_mode)
                
                if result["found"]:
                    if "matched_question" in result:
                        print(f"\nMatched Question: {result['matched_question']}")
                    if "confidence" in result:
                        print(f"Confidence: {result['confidence']:.4f}")
                    if "source" in result:
                        print(f"Source: {result['source']}")
                    if "reasoning" in result:
                        print(f"Reasoning: {result['reasoning']}")
                    print("\nAnswer:")
                    print(result['answer'])
                    
                    # Display debug info if requested
                    if debug_mode and "candidates" in result:
                        print("\n--- Debug Info ---")
                        print("\nCandidate Questions:")
                        for i, candidate in enumerate(result["candidates"]):
                            print(f"{i+1}. {candidate['question']} (Score: {candidate['score']:.4f})")
                        
                        if "gpt_response" in result:
                            print("\nGPT Analysis:")
                            print(result["gpt_response"])
                else:
                    print(result.get("message", "No suitable answer found. Please rephrase your question."))
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Exiting...")
                exit_requested = True
            except Exception as e:
                print(f"\nAn error occurred: {e}")
    
    except Exception as e:
        print(f"\nAn error occurred during main execution: {e}")


def main():
    """Synchronous wrapper for main_async"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupt received, shutting down gracefully.")
    except asyncio.CancelledError:
        # This is expected when properly exiting
        pass
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
