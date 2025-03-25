import os
import time
import json
import hashlib
import pickle
import asyncio
import logging
import concurrent.futures
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from functools import partial
from openai import OpenAI, AsyncOpenAI
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
FAQ_DIR = "."  # Files are in the current directory
FAQ_FILES = [
    "payroll.txt",
    "nrb_banking.txt",
    "retails_products.txt",
    "women_banking.txt",
    "sme_banking.txt"
]
VECTOR_STORE_NAME = "bengali_faq_store"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.05"))
MAX_NUM_RESULTS = 5  # Max number of search results to return
CACHE_DIR = "cache"  # Directory for caching
VECTOR_CACHE_FILE = os.path.join(CACHE_DIR, "vector_cache.pkl")
VECTOR_STORE_ID_FILE = os.path.join(CACHE_DIR, "vector_store_id.txt")
FILE_HASH_CACHE = os.path.join(CACHE_DIR, "file_hashes.json")
FILE_IDS_CACHE = os.path.join(CACHE_DIR, "file_ids.json")
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Initial delay between retries in seconds
MAX_CONCURRENT_UPLOADS = 5  # Maximum number of concurrent file uploads
MAX_CONCURRENT_FILE_ADDS = 5  # Maximum number of concurrent file additions to vector store
MAX_WORKERS = 4  # Maximum number of thread workers for file processing


class AsyncRAGSystem:
    """Enhanced Bengali FAQ Retrieval-Augmented Generation System using OpenAI API with asyncio"""
    
    def __init__(self):
        self.vector_store_id = None
        self.file_ids = []
        self.file_info = {}
        self.cache = {}
        self.sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Load cached data if available
        self._load_cache()
    
    def _load_cache(self):
        """Load cached vector store ID and file information"""
        # Load vector store ID if exists
        if os.path.exists(VECTOR_STORE_ID_FILE):
            try:
                with open(VECTOR_STORE_ID_FILE, 'r') as f:
                    self.vector_store_id = f.read().strip()
                logger.info(f"Loaded cached vector store ID: {self.vector_store_id}")
            except Exception as e:
                logger.error(f"Error loading cached vector store ID: {e}")
        
        # Load file hashes
        if os.path.exists(FILE_HASH_CACHE):
            try:
                with open(FILE_HASH_CACHE, 'r') as f:
                    self.cache['file_hashes'] = json.load(f)
                logger.info(f"Loaded {len(self.cache['file_hashes'])} cached file hashes")
            except Exception as e:
                logger.error(f"Error loading cached file hashes: {e}")
                self.cache['file_hashes'] = {}
        else:
            self.cache['file_hashes'] = {}
        
        # Load file IDs
        if os.path.exists(FILE_IDS_CACHE):
            try:
                with open(FILE_IDS_CACHE, 'r') as f:
                    self.file_ids = json.load(f)
                logger.info(f"Loaded {len(self.file_ids)} cached file IDs")
            except Exception as e:
                logger.error(f"Error loading cached file IDs: {e}")
        
        # Load vector cache if exists
        if os.path.exists(VECTOR_CACHE_FILE):
            try:
                with open(VECTOR_CACHE_FILE, 'rb') as f:
                    self.cache['vectors'] = pickle.load(f)
                logger.info(f"Loaded vector cache with {len(self.cache['vectors'])} entries")
            except Exception as e:
                logger.error(f"Error loading vector cache: {e}")
                self.cache['vectors'] = {}
        else:
            self.cache['vectors'] = {}
    
    def _save_cache(self):
        """Save current state to cache"""
        # Save vector store ID
        if self.vector_store_id:
            try:
                with open(VECTOR_STORE_ID_FILE, 'w') as f:
                    f.write(self.vector_store_id)
            except Exception as e:
                logger.error(f"Error saving vector store ID: {e}")
        
        # Save file hashes
        try:
            with open(FILE_HASH_CACHE, 'w') as f:
                json.dump(self.cache.get('file_hashes', {}), f)
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")
        
        # Save file IDs
        try:
            with open(FILE_IDS_CACHE, 'w') as f:
                json.dump(self.file_ids, f)
        except Exception as e:
            logger.error(f"Error saving file IDs: {e}")
        
        # Save vector cache
        try:
            with open(VECTOR_CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache.get('vectors', {}), f)
        except Exception as e:
            logger.error(f"Error saving vector cache: {e}")
        
        logger.info(f"Cache saved successfully: {len(self.file_ids)} file IDs, {len(self.cache.get('vectors', {}))} vector entries")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
                return hashlib.md5(file_data).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def _check_for_updates(self) -> Tuple[bool, Set[str], Set[str]]:
        """Check if FAQ files have been modified since last run
        
        Returns:
            Tuple[bool, Set[str], Set[str]]: (any_updates, new_files, modified_files)
        """
        new_files = set()
        modified_files = set()
        
        for filename in FAQ_FILES:
            filepath = os.path.join(FAQ_DIR, filename)
            if not os.path.exists(filepath):
                continue
                
            current_hash = self._calculate_file_hash(filepath)
            if current_hash:
                # Check if file is new or modified
                if filename not in self.cache.get('file_hashes', {}):
                    new_files.add(filename)
                    self.cache.setdefault('file_hashes', {})[filename] = current_hash
                elif self.cache['file_hashes'][filename] != current_hash:
                    modified_files.add(filename)
                    self.cache['file_hashes'][filename] = current_hash
        
        return (len(new_files) > 0 or len(modified_files) > 0), new_files, modified_files
    
    def _preprocess_faq_file(self, filepath: str) -> List[Dict[str, str]]:
        """Preprocess FAQ file to extract Q&A pairs with special handling for Bengali text"""
        try:
            # Ensure proper UTF-8 encoding for Bengali text
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file might contain Bengali text
            has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in content)
            if has_bengali:
                logger.info(f"Bengali text detected in {filepath}")
            
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
                            "answer": ' '.join(current_answer)
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
                    "answer": ' '.join(current_answer)
                })
            
            # Debug output to verify extraction (limiting display length for Bengali text)
            logger.info(f"Extracted {len(pairs)} Q&A pairs from {filepath}")
            
            return pairs
            
        except UnicodeDecodeError as ue:
            logger.error(f"Unicode error processing {filepath} - possible Bengali encoding issue: {ue}")
            # Try with a different encoding as fallback
            try:
                with open(filepath, 'r', encoding='utf-16') as f:
                    content = f.read()
                logger.info(f"Successfully read file with utf-16 encoding")
                # Recursively call the function with the new content
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
                        "answer": ' '.join(current_answer)
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
                "answer": ' '.join(current_answer)
            })
        
        logger.info(f"Extracted {len(pairs)} Q&A pairs from content")
        return pairs
    
    async def _upload_qa_pair_async(self, pair: Dict[str, str], filename: str, index: int) -> Optional[str]:
        """Upload a single Q&A pair to OpenAI and return the file ID (async version)"""
        temp_filepath = f"temp_{filename}_{index}.txt"
        
        try:
            content = f"Source: {filename}\nQuestion: {pair['question']}\nAnswer: {pair['answer']}"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            for retry in range(MAX_RETRIES):
                try:
                    with open(temp_filepath, 'rb') as f:
                        response = await self.client.files.create(
                            file=f,
                            purpose="user_data"  # Using user_data for vector store usage
                        )
                    logger.info(f"Uploaded Q&A pair {index} from {filename} as file ID: {response.id}")
                    return response.id
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** retry)  # Exponential backoff
                        logger.warning(f"Upload attempt {retry+1} failed for {filename}:{index}, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All upload attempts failed for Q&A pair {index} from {filename}: {e}")
                        return None
        except Exception as e:
            logger.error(f"Error preparing Q&A pair {index} from {filename} for upload: {e}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    def _upload_qa_pair_sync(self, pair: Dict[str, str], filename: str, index: int) -> Optional[str]:
        """Synchronous version of upload_qa_pair for use with ThreadPoolExecutor"""
        temp_filepath = f"temp_{filename}_{index}.txt"
        
        try:
            content = f"Source: {filename}\nQuestion: {pair['question']}\nAnswer: {pair['answer']}"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            for retry in range(MAX_RETRIES):
                try:
                    with open(temp_filepath, 'rb') as f:
                        response = self.sync_client.files.create(
                            file=f,
                            purpose="user_data"
                        )
                    logger.info(f"Uploaded Q&A pair {index} from {filename} as file ID: {response.id}")
                    return response.id
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** retry)
                        logger.warning(f"Upload attempt {retry+1} failed for {filename}:{index}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All upload attempts failed for Q&A pair {index} from {filename}: {e}")
                        return None
        except Exception as e:
            logger.error(f"Error preparing Q&A pair {index} from {filename} for upload: {e}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    async def _get_or_create_vector_store_async(self) -> Optional[str]:
        """Get existing vector store or create a new one (async version)"""
        try:
            # If we already have a cached vector store ID, verify it still exists
            if self.vector_store_id:
                try:
                    # Try to retrieve the vector store to verify it exists
                    vector_store = await self.client.vector_stores.retrieve(vector_store_id=self.vector_store_id)
                    logger.info(f"Using cached vector store: {self.vector_store_id}")
                    return self.vector_store_id
                except Exception as e:
                    logger.warning(f"Cached vector store {self.vector_store_id} not found or error: {e}")
                    # Continue to create a new one
            
            # List existing vector stores
            vector_stores = await self.client.vector_stores.list(limit=100)
            for store in vector_stores.data:
                if store.name == VECTOR_STORE_NAME:
                    logger.info(f"Using existing vector store: {store.id}")
                    self.vector_store_id = store.id
                    return store.id
            
            # Create a new vector store
            response = await self.client.vector_stores.create(
                name=VECTOR_STORE_NAME,
            )
            logger.info(f"Created new vector store: {response.id}")
            self.vector_store_id = response.id
            return response.id
        except Exception as e:
            logger.error(f"Error getting/creating vector store: {e}")
            return None
    
    async def _add_file_to_vector_store_async(self, file_id: str) -> bool:
        """Add a file to the vector store (async version)"""
        if not self.vector_store_id:
            logger.error("Vector store ID is not set")
            return False
        
        for retry in range(MAX_RETRIES):
            try:
                # Create a file in the vector store
                await self.client.vector_stores.files.create(
                    vector_store_id=self.vector_store_id,
                    file_id=file_id
                )
                
                logger.info(f"File {file_id} added to vector store successfully")
                return True
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** retry)
                    logger.warning(f"Add to vector store attempt {retry+1} failed for file {file_id}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All attempts to add file {file_id} to vector store failed: {e}")
                    return False
    
    def _add_file_to_vector_store_sync(self, file_id: str) -> bool:
        """Synchronous version of add_file_to_vector_store for use with ThreadPoolExecutor"""
        if not self.vector_store_id:
            logger.error("Vector store ID is not set")
            return False
        
        for retry in range(MAX_RETRIES):
            try:
                # Create a file in the vector store
                self.sync_client.vector_stores.files.create(
                    vector_store_id=self.vector_store_id,
                    file_id=file_id
                )
                
                logger.info(f"File {file_id} added to vector store successfully")
                return True
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** retry)
                    logger.warning(f"Add to vector store attempt {retry+1} failed for file {file_id}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts to add file {file_id} to vector store failed: {e}")
                    return False
    
    async def _process_faq_file_async(self, filename: str, files_to_process: set) -> List[str]:
        """Process a single FAQ file, extract Q&A pairs, and upload them async"""
        filepath = os.path.join(FAQ_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Warning: File {filepath} does not exist. Skipping.")
            return []
        
        # Check if we have this file's Q&A pairs cached
        file_key = f"{filename}_{self._calculate_file_hash(filepath)}"
        cached_pairs = self.cache.get('vectors', {}).get(file_key, None)
        
        if cached_pairs:
            logger.info(f"Using {len(cached_pairs)} cached Q&A pairs for {filename}")
            faq_pairs = cached_pairs
        else:
            # Extract new Q&A pairs
            faq_pairs = self._preprocess_faq_file(filepath)
            if not faq_pairs:
                logger.warning(f"Warning: No FAQ pairs extracted from {filepath}. Skipping.")
                return []
            
            # Cache the extracted pairs
            self.cache.setdefault('vectors', {})[file_key] = faq_pairs
        
        # Process each Q&A pair concurrently
        file_ids = []
        upload_tasks = []
        
        # Upload Q&A pairs in batches to avoid overwhelming the API
        for i, pair in enumerate(faq_pairs):
            upload_task = self._upload_qa_pair_async(pair, filename, i)
            upload_tasks.append(upload_task)
            
            # Process in batches of MAX_CONCURRENT_UPLOADS
            if len(upload_tasks) >= MAX_CONCURRENT_UPLOADS:
                completed_uploads = await asyncio.gather(*upload_tasks)
                file_ids.extend([fid for fid in completed_uploads if fid])
                upload_tasks = []
        
        # Process any remaining uploads
        if upload_tasks:
            completed_uploads = await asyncio.gather(*upload_tasks)
            file_ids.extend([fid for fid in completed_uploads if fid])
        
        return file_ids
    
    async def initialize_async(self) -> bool:
        """Initialize the RAG system async version"""
        try:
            logger.info("Initializing Bengali FAQ RAG system...")
            
            # Check if FAQ directory exists
            if not os.path.exists(FAQ_DIR):
                logger.error(f"Error: FAQ directory '{FAQ_DIR}' does not exist.")
                logger.info(f"Creating the directory... Please add your FAQ files there.")
                os.makedirs(FAQ_DIR, exist_ok=True)
                return False
            
            # Check if we need to update our vector store
            updates_needed, new_files, modified_files = self._check_for_updates()
            
            # If we have a cached vector store and no updates needed, use it
            if self.vector_store_id and not updates_needed and self.file_ids:
                logger.info(f"Using cached vector store (no updates detected): {self.vector_store_id}")
                return True
            
            # Get or create vector store
            self.vector_store_id = await self._get_or_create_vector_store_async()
            if not self.vector_store_id:
                logger.error("Failed to get or create vector store. Aborting initialization.")
                return False
            
            # Track which files we've already processed
            files_to_process = set()
            
            # Determine which files need processing
            if updates_needed and self.vector_store_id:
                # Process only new or modified files
                for filename in new_files.union(modified_files):
                    files_to_process.add(filename)
                logger.info(f"Processing {len(new_files)} new and {len(modified_files)} modified files")
            else:
                # Process all files (full initialization)
                for filename in FAQ_FILES:
                    if os.path.exists(os.path.join(FAQ_DIR, filename)):
                        files_to_process.add(filename)
                logger.info(f"Processing all {len(files_to_process)} FAQ files")
            
            # Process all FAQ files concurrently
            file_processing_tasks = [
                self._process_faq_file_async(filename, files_to_process)
                for filename in files_to_process
            ]
            
            # Wait for all files to be processed
            all_file_ids_nested = await asyncio.gather(*file_processing_tasks)
            all_file_ids = [fid for sublist in all_file_ids_nested for fid in sublist]
            
            # Update file IDs list with new IDs
            self.file_ids.extend(all_file_ids)
            
            # Add all files to vector store concurrently
            add_tasks = []
            for file_id in all_file_ids:
                add_tasks.append(self._add_file_to_vector_store_async(file_id))
                
                # Process in batches to avoid overwhelming the API
                if len(add_tasks) >= MAX_CONCURRENT_FILE_ADDS:
                    await asyncio.gather(*add_tasks)
                    add_tasks = []
            
            # Process any remaining tasks
            if add_tasks:
                await asyncio.gather(*add_tasks)
            
            # Update our cache
            self._save_cache()
            
            logger.info(f"Initialized RAG system with {len(self.file_ids)} Q&A pairs")
            
            return len(self.file_ids) > 0
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            # Clean up any partially created resources if initialization fails
            await self.cleanup_async(preserve_cache=True, preserve_resources=True)
            return False
    
    async def answer_query_async(self, query: str, return_best_even_if_below_threshold: bool = True, debug_mode: bool = False) -> Dict:
        """Answer a user query by directly searching the vector store and returning verbatim answers (async version)"""
        try:
            if not self.vector_store_id:
                return {"found": False, "message": "RAG system not initialized."}
            
            # Check if query is in Bengali
            has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
            if has_bengali:
                logger.info(f"Processing Bengali query: {query}")
            
            # Directly search the vector store for the most semantically similar content
            search_results = await self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
                max_num_results=MAX_NUM_RESULTS
            )
            
            # Process search results
            if not search_results.data or len(search_results.data) == 0:
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
            
            # Get the best match
            best_match = search_results.data[0]
            match_score = best_match.score
            
            # Check if score meets the threshold
            if match_score < SIMILARITY_THRESHOLD and not return_best_even_if_below_threshold:
                if has_bengali:
                    return {
                        "found": False,
                        "message": f"কোন উপযুক্ত মিল পাওয়া যায়নি (সর্বোচ্চ স্কোর: {match_score:.4f})। অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লিখুন।"
                    }
                else:
                    return {
                        "found": False,
                        "message": f"No suitable match found (best score: {match_score:.4f}). Please rephrase your question."
                    }
            
            # Extract verbatim question and answer from the matched content
            matched_question = ""
            matched_answer = ""
            source_file = best_match.filename
            
            # Define markers in both English and Bengali
            q_markers = ["Question:", "প্রশ্ন:"]  # English and Bengali
            a_markers = ["Answer:", "উত্তর:"]     # English and Bengali
            
            for chunk in best_match.content:
                chunk_text = chunk.text
                
                # Try each combination of question and answer markers
                for q_marker in q_markers:
                    for a_marker in a_markers:
                        q_index = chunk_text.find(q_marker)
                        a_index = chunk_text.find(a_marker)
                        
                        if q_index != -1 and a_index != -1:
                            # Extract the verbatim question and answer
                            matched_question = chunk_text[q_index + len(q_marker):a_index].strip()
                            matched_answer = chunk_text[a_index + len(a_marker):].strip()
                            break
                    
                    # If we've found a match, break the outer loop too
                    if matched_question and matched_answer:
                        break
                
                # If we've found a match, break the content chunks loop
                if matched_question and matched_answer:
                    break
            
            # If we couldn't find using markers, try to extract from source content
            if not matched_question or not matched_answer:
                # The content might not have explicit markers, try to extract from the structure
                # First, look for "Source:" which our upload format includes
                source_index = -1
                for chunk in best_match.content:
                    chunk_text = chunk.text
                    source_index = chunk_text.find("Source:")
                    if source_index != -1:
                        # Try to find the content after "Source:" line
                        lines = chunk_text[source_index:].split('\n')
                        if len(lines) >= 3:  # Source, Question, Answer
                            for i, line in enumerate(lines):
                                if i > 0 and any(marker in line for marker in q_markers):
                                    q_line = line
                                    if i+1 < len(lines) and any(marker in lines[i+1] for marker in a_markers):
                                        a_line = lines[i+1]
                                        # Extract content after the markers
                                        for q_marker in q_markers:
                                            if q_marker in q_line:
                                                matched_question = q_line.split(q_marker, 1)[1].strip()
                                                break
                                        for a_marker in a_markers:
                                            if a_marker in a_line:
                                                matched_answer = a_line.split(a_marker, 1)[1].strip()
                                                if i+2 < len(lines):  # Include additional lines in answer
                                                    for j in range(i+2, len(lines)):
                                                        if not any(marker in lines[j] for marker in q_markers + a_markers):
                                                            matched_answer += " " + lines[j].strip()
                                                break
                                        break
                            break
            
            # Prepare result
            result = {
                "found": True,
                "matched_question": matched_question,
                "answer": matched_answer,
                "confidence": match_score,
                "source_file": source_file
            }
            
            # Add all matches for debug mode
            if debug_mode:
                all_matches = []
                for match in search_results.data:
                    match_data = {
                        "score": match.score,
                        "filename": match.filename,
                        "file_id": match.file_id
                    }
                    
                    # Extract question and answer for debugging
                    for chunk in match.content:
                        chunk_text = chunk.text
                        found_q = False
                        found_a = False
                        
                        for q_marker in q_markers:
                            q_index = chunk_text.find(q_marker)
                            if q_index != -1:
                                for a_marker in a_markers:
                                    a_index = chunk_text.find(a_marker)
                                    if a_index != -1 and a_index > q_index:
                                        match_data["question"] = chunk_text[q_index + len(q_marker):a_index].strip()
                                        match_data["answer"] = chunk_text[a_index + len(a_marker):].strip()
                                        found_q = found_a = True
                                        break
                            if found_q and found_a:
                                break
                        if found_q and found_a:
                            break
                    
                    all_matches.append(match_data)
                
                result["all_matches"] = all_matches
            
            # Check if we have properly extracted question and answer
            if not matched_question or not matched_answer:
                logger.warning("Failed to extract proper question and answer from matched content")
                if debug_mode:
                    # Include the raw content for debugging
                    raw_content = []
                    for chunk in best_match.content:
                        raw_content.append(chunk.text)
                    result["raw_content"] = raw_content
            
            return result
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return {"found": False, "message": f"Error processing query: {str(e)}"}

    def answer_query(self, query: str, return_best_even_if_below_threshold: bool = True, debug_mode: bool = False) -> Dict:
        """Synchronous wrapper for answer_query_async"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.answer_query_async(query, return_best_even_if_below_threshold, debug_mode)
        )
    
    async def cleanup_async(self, preserve_cache=True, preserve_resources=True):
        """Clean up resources but preserve them by default (async version)
        
        Args:
            preserve_cache: If True, keep the cached vector store ID and other cache items
            preserve_resources: If True, keep the actual files and vector store
        """
        # If preserve_resources is True, we don't delete anything from the OpenAI API
        if not preserve_resources:
            try:
                # Delete individual files from the vector store
                for file_id in self.file_ids:
                    try:
                        if self.vector_store_id:
                            try:
                                await self.client.vector_stores.files.delete(
                                    vector_store_id=self.vector_store_id,
                                    file_id=file_id
                                )
                                logger.info(f"Removed file {file_id} from vector store")
                            except Exception as e:
                                logger.error(f"Error removing file {file_id} from vector store: {e}")
                        
                        # Then delete the file itself
                        try:
                            await self.client.files.delete(file_id=file_id)
                            logger.info(f"Deleted file: {file_id}")
                        except Exception as e:
                            logger.error(f"Error deleting file {file_id}: {e}")
                    except Exception as e:
                        logger.error(f"Error handling file cleanup for {file_id}: {e}")
                
                # Delete the vector store if we're not preserving the cache
                if self.vector_store_id and not preserve_cache:
                    try:
                        await self.client.vector_stores.delete(vector_store_id=self.vector_store_id)
                        logger.info(f"Deleted vector store: {self.vector_store_id}")
                        
                        # Clear cached vector store ID
                        self.vector_store_id = None
                        if os.path.exists(VECTOR_STORE_ID_FILE):
                            os.remove(VECTOR_STORE_ID_FILE)
                            logger.info("Removed cached vector store ID")
                    except Exception as e:
                        logger.error(f"Error deleting vector store {self.vector_store_id}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning up resources: {e}")
        
        # Clear file IDs list if we're not preserving the cache
        if not preserve_cache:
            self.file_ids = []
        
        if preserve_cache:
            # Only save if we're preserving the cache
            self._save_cache()
            logger.info("Preserved cache for future use")
    
    def cleanup(self, preserve_cache=True, preserve_resources=True):
        """Synchronous wrapper for cleanup_async"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.cleanup_async(preserve_cache, preserve_resources)
        )
    
    async def clear_cache_async(self):
        """Clear all cached data (async version)"""
        try:
            # Clear in-memory cache
            self.cache = {}
            
            # Delete cache files
            for cache_file in [VECTOR_CACHE_FILE, VECTOR_STORE_ID_FILE, FILE_HASH_CACHE, FILE_IDS_CACHE]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def clear_cache(self):
        """Synchronous wrapper for clear_cache_async"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.clear_cache_async())
    
    def initialize(self) -> bool:
        """Synchronous wrapper for initialize_async"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.initialize_async())


async def main_async():
    # Initialize the RAG system
    rag = AsyncRAGSystem()
    initialization_success = await rag.initialize_async()
    
    try:
        # Interactive query testing
        print("\nBengali FAQ RAG System")
        print("Type 'exit' to quit")
        print("Type 'debug on' to enable debug mode")
        print("Type 'debug off' to disable debug mode")
        print("Type 'clear cache' to clear all cached data")
        
        if not initialization_success:
            print("\nWARNING: RAG system was not initialized properly.")
            print("Queries will likely fail. Please check error messages above.")
            print("Make sure your FAQ files are in the 'faq_data' directory.")
            print("You can still try queries, but they may not work correctly.")
        
        debug_mode = False
        
        while True:
            query = input("\nEnter your query in Bengali (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            elif query.lower() == 'debug on':
                debug_mode = True
                print("Debug mode enabled. Will show all potential matches.")
                continue
            elif query.lower() == 'debug off':
                debug_mode = False
                print("Debug mode disabled.")
                continue
            elif query.lower() == 'clear cache':
                await rag.clear_cache_async()
                continue
            
            result = await rag.answer_query_async(query, return_best_even_if_below_threshold=True, debug_mode=debug_mode)
            
            if result["found"]:
                if "matched_question" in result:
                    print(f"\nMatched Question: {result['matched_question']}")
                if "confidence" in result:
                    print(f"Confidence: {result['confidence']:.4f}")
                if "source_file" in result:
                    print(f"Source: {result['source_file']}")
                print("\nAnswer:")
                print(result['answer'])
            else:
                print(result.get("message", "No suitable answer found. Please rephrase your question."))
            
            # Display all potential matches in debug mode
            if debug_mode and "all_matches" in result:
                print("\n--- All Potential Matches ---")
                for i, match in enumerate(result["all_matches"]):
                    print(f"\nMatch {i+1} (Score: {match.get('score', 0):.4f}, File: {match.get('filename', 'Unknown')})")
                    if "question" in match:
                        print(f"Question: {match['question']}")
                    if "answer" in match:
                        answer_preview = match['answer']
                        if len(answer_preview) > 100:
                            answer_preview = answer_preview[:100] + "..."
                        print(f"Answer: {answer_preview}")
    
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources (preserve cache and resources by default)
        print("\nCleaning up resources...")
        await rag.cleanup_async(preserve_cache=True, preserve_resources=True)


def main():
    """Synchronous wrapper for main_async"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())


if __name__ == "__main__":
    main()
