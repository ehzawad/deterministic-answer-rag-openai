#!/usr/bin/env python3
"""
Advanced Bengali FAQ System with Ultra-Precision Hybrid Matching

This module provides a production-ready Bengali FAQ system with:
- Deterministic answer retrieval with zero hallucination
- File-as-cluster architecture for cross-domain disambiguation
- Hybrid scoring: embeddings + n-grams + keywords + semantic boosts
- Bengali text normalization and cleaning
- ChromaDB vector storage with corruption recovery
- Performance optimizations and caching
"""

import os
import sys
import json
import hashlib
import asyncio
import traceback
import re
import logging
import time
import glob
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, timedelta

# Third-party imports with error handling
try:
    from openai import OpenAI
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install openai chromadb")
    sys.exit(1)

from hybrid_matcher import HybridMatcher, hybrid_enhance_candidates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BengaliFAQ-Service')

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.query_times = []
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0
        
    def record_query_time(self, duration: float):
        self.query_times.append(duration)
        # Keep only last 1000 queries
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]
    
    def get_avg_query_time(self) -> float:
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0.0
    
    def get_cache_hit_rate(self) -> float:
        total = self.embedding_cache_hits + self.embedding_cache_misses
        return self.embedding_cache_hits / total if total > 0 else 0.0

performance_monitor = PerformanceMonitor()

# Connection pool for OpenAI
class OpenAIConnectionPool:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key, max_retries=max_retries)
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    async def create_embeddings_async(self, texts: List[str], model: str) -> List[List[float]]:
        """Rate-limited async embedding creation"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            input=texts,
            model=model
        )
        
        self.last_request_time = time.time()
        return [embedding.embedding for embedding in response.data]

# Embedding cache with TTL
class EmbeddingCache:
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[List[List[float]], datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _hash_key(self, texts: List[str]) -> str:
        """Create a hash key for the text list"""
        combined = "|".join(sorted(texts))
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get cached embeddings if available and not expired"""
        key = self._hash_key(texts)
        if key in self.cache:
            embeddings, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                performance_monitor.embedding_cache_hits += 1
                return embeddings
            else:
                # Expired, remove from cache
                del self.cache[key]
        
        performance_monitor.embedding_cache_misses += 1
        return None
    
    def set(self, texts: List[str], embeddings: List[List[float]]):
        """Cache embeddings with timestamp"""
        key = self._hash_key(texts)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (embeddings, datetime.now())
    
    def clear_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

embedding_cache = EmbeddingCache()

# Load configuration
def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        # Fallback configuration
        return {
            "models": {
                "embedding_model": "text-embedding-3-large",
                "core_model": "gpt-4.1-nano"
            },
            "system": {
                "confidence_threshold": 0.9,
                "max_candidates": 1,
                "embedding_dimensions": 1024
            },
            "directories": {
                "faq_dir": "faq_data",
                "cache_dir": "cache"
            },
            "logging": {
                "level": "INFO"
            }
        }

# Load configuration
config = load_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BengaliFAQ-Service')

# Constants from config
FAQ_DIR = config['directories']['faq_dir']
CACHE_DIR = config['directories']['cache_dir']
CHROMA_DB_DIR = os.path.join(CACHE_DIR, "chroma_db")
FILE_HASH_CACHE = os.path.join(CACHE_DIR, "file_hashes.json")
CONFIDENCE_THRESHOLD = config['system']['confidence_threshold']
MAX_CANDIDATES = config['system']['max_candidates']
EMBEDDING_MODEL = config.get('embedding_model', 'text-embedding-3-large')
EMBEDDING_DIMENSIONS = config['system']['embedding_dimensions']

# Prime words for routing queries to specific collections
PRIME_WORDS = {
    "yaqeen": ["ইয়াকিন", "ইয়াকি", "ইসলামিক", "শরিয়া", "মুদারাবা", "উজরা", "হালাল", "মুরাবাহা", "বাই", "অঘনিয়া", "আগানিয়া"],
    "sme": ["এসএমই", "ব্যবসা", "উদ্যোক্তা", "কৃষি", "ব্যবসায়ী", "প্রবাহ", "বুনিয়াদ"],
    "retail": ["রেগুলার", "ইন্সপায়ার", "সিনিয়র", "জুনিয়র", "লাখপতি", "কোটিপতি", "মিলিয়নিয়ার", "ব্রিক"],
    "card": ["কার্ড", "ক্রেডিট", "ডেবিট", "ভিসা", "মাস্টার", "রিওয়ার্ড", "প্রিপেইড"],
    "women": ["অঙ্গনা", "নারী", "মহিলা", "বানাত"],
    "payroll": ["পেরোল", "পেয়রোল", "বেতন", "স্যালারি"],
    "privilege": ["প্রিভিলেজ", "বিশেষাধিকার"],
    "agent": ["এজেন্ট", "প্রতিনিধি"],
    "nrb": ["এনআরবি", "প্রবাসী", "রেমিটেন্স"]
}

# File mapping to collections
FILE_TO_COLLECTION = {
    "yaqeen.txt": "yaqeen",
    "sme_banking.txt": "sme", 
    "retails_products.txt": "retail",
    "card_faqs.txt": "card",
    "women_banking.txt": "women",
    "payroll.txt": "payroll",
    "Privilege_faqs.txt": "privilege",
    "agent_banking.txt": "agent",
    "nrb_banking.txt": "nrb"
}

# Import the advanced Bengali semantic engine
from bengali_semantic_engine import BengaliSemanticEngine

class BengaliFAQService:
    """
    🇧🇩 Advanced Bengali FAQ Service with Deep Semantic Understanding
    Enhanced with sophisticated linguistic processing for nuanced Bengali queries
    """
    
    def __init__(self):
        # Check for test mode flag
        self.test_mode = os.getenv('FAQ_TEST_MODE', '').lower() == 'true'
        
        # Handle API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not self.test_mode:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            if self.test_mode:
                logger.info("Running in test mode (FAQ_TEST_MODE=true)")
            else:
                logger.warning("OPENAI_API_KEY not found. Running in test mode (no embeddings).")
        
        # Initialize core attributes
        self.file_hashes = {}
        self.initialized = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Initialize hybrid matcher
        self.hybrid_matcher = HybridMatcher(config)
        
        # 🧠 NEW: Initialize advanced Bengali semantic engine
        self.bengali_semantic_engine = BengaliSemanticEngine()
        
        # Load file hashes
        self._load_file_hashes()
        
        # Auto-initialize if not in test mode
        try:
            if not self.test_mode:
                logger.info("Auto-initializing Bengali FAQ Service...")
                self.initialize()
        except Exception as e:
            logger.warning(f"Could not auto-initialize: {e}")
    
    def _load_file_hashes(self):
        """Load cached file hashes"""
        if os.path.exists(FILE_HASH_CACHE):
            try:
                with open(FILE_HASH_CACHE, 'r') as f:
                    self.file_hashes = json.load(f)
                logger.info(f"Loaded {len(self.file_hashes)} cached file hashes")
            except Exception as e:
                logger.error(f"Error loading cached file hashes: {e}")
                self.file_hashes = {}
    
    def _save_file_hashes(self):
        """Save file hashes to cache"""
        try:
            with open(FILE_HASH_CACHE, 'w') as f:
                json.dump(self.file_hashes, f)
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")
    
    def _discover_faq_files(self) -> List[str]:
        """Dynamically discover all .txt files in the FAQ directory"""
        try:
            if not os.path.exists(FAQ_DIR):
                logger.warning(f"FAQ directory '{FAQ_DIR}' does not exist.")
                return []
            
            txt_files = glob.glob(os.path.join(FAQ_DIR, "*.txt"))
            filenames = [os.path.basename(filepath) for filepath in txt_files]
            
            logger.info(f"Discovered {len(filenames)} .txt files in {FAQ_DIR}: {filenames}")
            return filenames
        except Exception as e:
            logger.error(f"Error discovering FAQ files: {e}")
            return []
    
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
        """Check if FAQ files have been modified, added, or deleted since last run"""
        files_to_process = set()
        
        # Discover current files on disk
        discovered_files = set(self._discover_faq_files())
        # Get files known from the last run (from cache)
        known_files = set(self.file_hashes.keys())
        
        # 1. Find new and modified files
        for filename in discovered_files:
            filepath = os.path.join(FAQ_DIR, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Warning: File {filepath} does not exist (should not happen).")
                continue
                
            current_hash = self._calculate_file_hash(filepath)
            if current_hash:
                # If file is new or hash has changed, mark for processing
                if filename not in self.file_hashes or self.file_hashes[filename] != current_hash:
                    files_to_process.add(filename)
                    self.file_hashes[filename] = current_hash
        
        # 2. Find deleted files
        deleted_files = known_files - discovered_files
        if deleted_files:
            logger.info(f"Detected {len(deleted_files)} deleted FAQ files: {deleted_files}")
            for filename in deleted_files:
                # Remove the collection associated with the deleted file
                collection_name = self._get_collection_name(filename)
                try:
                    self.chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Deleted orphaned collection: {collection_name}")
                except Exception as e:
                    # An exception might be raised if collection doesn't exist for some reason
                    logger.warning(f"Could not delete collection '{collection_name}': {e}")
                
                # Remove from our file hash tracking
                del self.file_hashes[filename]
        
        # A change is either a file to process or a file that was deleted
        needs_update = len(files_to_process) > 0 or len(deleted_files) > 0
        
        return needs_update, files_to_process
    
    def _preprocess_faq_file(self, filepath: str) -> List[Dict[str, str]]:
        """Preprocess FAQ file to extract Q&A pairs with Bengali text handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            q_markers = ["Question:", "প্রশ্ন:"]
            a_markers = ["Answer:", "উত্তর:"]
            
            pairs = []
            current_question = None
            current_answer = []
            in_answer = False
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if not line:
                    # Treat blank lines as a separator between Q&A pairs
                    if current_question and current_answer:
                        clean_q = self._clean_text(current_question)
                        clean_a = ' '.join(current_answer)
                        if clean_q and clean_a:
                            pairs.append({
                                "question": clean_q,
                                "answer": clean_a,
                                "source": os.path.basename(filepath)
                            })
                        # Reset for the next pair
                        current_question = None
                        current_answer = []
                        in_answer = False
                    continue
                
                is_question_line = any(line.startswith(marker) for marker in q_markers)
                is_answer_line = any(line.startswith(marker) for marker in a_markers)
                
                if is_question_line:
                    if current_question and current_answer:
                        clean_q = self._clean_text(current_question)
                        clean_a = ' '.join(current_answer)
                        if clean_q and clean_a:
                            pairs.append({
                                "question": clean_q,
                                "answer": clean_a,
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
                clean_q = self._clean_text(current_question)
                clean_a = ' '.join(current_answer)
                if clean_q and clean_a:
                    pairs.append({
                        "question": clean_q,
                        "answer": clean_a,
                        "source": os.path.basename(filepath)
                    })
            
            logger.info(f"Extracted {len(pairs)} Q&A pairs from {filepath}")
            return pairs
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Bengali text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common banking terms
        banking_terms = {
            'ATM': 'এটিএম',
            'atm': 'এটিএম',
            'ক্যাশ মেশিন': 'এটিএম',
            'অটো টেলার': 'এটিএম'
        }
        
        # Normalize common spelling variations and typos
        spelling_variations = {
            'ইয়াকি': 'ইয়াকিন',
            'আগানিয়া': 'অঘনিয়া',
            'এমটিবি': 'এমটিবি',
            'মুতুয়াল': 'মিউচুয়াল',
            'পেয়রোল': 'পেরোল',
            'অ্যাকাউট': 'অ্যাকাউন্ট',  # Fix typo in payroll FAQ
            'রেগুলার অ্যাকাউট': 'রেগুলার অ্যাকাউন্ট',
            'পেরোল অ্যাকাউট': 'পেরোল অ্যাকাউন্ট'
        }
        
        for eng_term, bengali_term in banking_terms.items():
            text = text.replace(eng_term, bengali_term)
            
        for variation, standard in spelling_variations.items():
            text = text.replace(variation, standard)
        
        # Remove account numbers (sequences of 8+ digits)
        text = re.sub(r'\b\d{8,}\b', '', text)
        
        return text.strip()
    
    def _create_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Create embeddings for a list of texts using configured embedding model with caching"""
        if not texts:
            return None
        
        # In test mode OR if client not initialized, return dummy embeddings
        if self.test_mode or not self.client:
            logger.info(f"Test/dummy mode: Creating {len(texts)} dummy embeddings")
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
        
        # Check cache first
        cached_embeddings = embedding_cache.get(texts)
        if cached_embeddings:
            logger.debug(f"Cache hit: Retrieved {len(texts)} embeddings from cache")
            return cached_embeddings
        
        try:
            logger.debug(f"Creating embeddings for {len(texts)} texts")
            start_time = time.time()
            
            response = self.client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            
            # Cache the results
            embedding_cache.set(texts, embeddings)
            
            duration = time.time() - start_time
            logger.debug(f"Embeddings created in {duration:.2f}s, cached for future use")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def _get_collection_name(self, filename: str) -> str:
        """Get ChromaDB collection name for a file"""
        collection_type = FILE_TO_COLLECTION.get(filename, "general")
        return f"faq_{collection_type}"
    
    def _update_collection(self, filename: str, faq_pairs: List[Dict[str, str]]):
        """Update ChromaDB collection for a specific file"""
        try:
            collection_name = self._get_collection_name(filename)
            
            # 🛠️ Check if collection exists before trying to delete it
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Check if collection exists first
                    existing_collections = [coll.name for coll in self.chroma_client.list_collections()]
                    if collection_name in existing_collections:
                        self.chroma_client.delete_collection(collection_name)
                        logger.info(f"Deleted existing collection: {collection_name}")
                    else:
                        logger.debug(f"Collection {collection_name} doesn't exist, skipping deletion")
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.warning(f"Could not delete collection {collection_name}: {e}")
                    else:
                        logger.debug(f"Retry {attempt + 1} deleting collection {collection_name}")
            
            # Create new collection
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"filename": filename}
            )
            
            if not faq_pairs:
                logger.warning(f"No FAQ pairs to add to collection {collection_name}")
                return
            
            # Prepare data for ChromaDB
            questions = [pair["question"] for pair in faq_pairs]
            embeddings = self._create_embeddings(questions)
            
            if len(embeddings) != len(questions):
                logger.error(f"Embedding count mismatch for {filename}")
                return
            
            # 🛠️ Safer batch addition - add in smaller chunks to prevent corruption
            batch_size = 50  # Smaller batches are more reliable
            
            for i in range(0, len(faq_pairs), batch_size):
                batch_end = min(i + batch_size, len(faq_pairs))
                batch_pairs = faq_pairs[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_questions = questions[i:batch_end]
                
                ids = [f"{filename}_{j}" for j in range(i, batch_end)]
                metadatas = [
                    {
                        "question": pair["question"],
                        "answer": pair["answer"], 
                        "source": pair["source"]
                    }
                    for pair in batch_pairs
                ]
                
                try:
                    collection.add(
                        ids=ids,
                        embeddings=batch_embeddings,
                        documents=batch_questions,
                        metadatas=metadatas
                    )
                    logger.debug(f"Added batch {i//batch_size + 1} to {collection_name}")
                except Exception as batch_error:
                    logger.error(f"Error adding batch to {collection_name}: {batch_error}")
                    raise  # Re-raise to trigger collection rebuild
            
            logger.info(f"Added {len(faq_pairs)} entries to collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error updating collection for {filename}: {e}")
            logger.error(traceback.format_exc())
    
    def _detect_prime_words(self, query: str) -> List[str]:
        """Detect prime words in query to determine target collections"""
        detected_collections = []
        query_lower = query.lower()
        
        for collection_type, prime_words in PRIME_WORDS.items():
            for prime_word in prime_words:
                if prime_word.lower() in query_lower:
                    detected_collections.append(collection_type)
                    break
        
        return list(set(detected_collections))  # Remove duplicates
    
    def _detect_prime_words_cached(self, query_lower: str) -> List[str]:
        """SPEED OPTIMIZED: Detect prime words using pre-lowercased query"""
        detected_collections = []
        
        for collection_type, prime_words in PRIME_WORDS.items():
            for prime_word in prime_words:
                if prime_word.lower() in query_lower:
                    detected_collections.append(collection_type)
                    break
        
        return list(set(detected_collections))  # Remove duplicates
    
    def _test_mode_search(self, collection, query: str, n_results: int) -> List[Dict]:
        """Simple text-based search for test mode"""
        try:
            # Get all documents from collection
            all_data = collection.get(include=["metadatas", "documents"])
            
            if not all_data['metadatas']:
                return []
            
            candidates = []
            query_lower = query.lower()
            
            for metadata, document in zip(all_data['metadatas'], all_data['documents']):
                # Simple text matching score
                question_lower = metadata["question"].lower()
                
                # Count word matches
                query_words = set(query_lower.split())
                question_words = set(question_lower.split())
                
                if not query_words:
                    continue
                
                # Calculate simple similarity score
                matches = len(query_words.intersection(question_words))
                similarity = matches / len(query_words) if query_words else 0
                
                # Boost exact substring matches
                if query_lower in question_lower:
                    similarity += 0.5
                
                # Cap at 1.0
                similarity = min(similarity, 1.0)
                
                if similarity > 0:  # Only include if there's some match
                    candidates.append({
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "source": metadata["source"],
                        "score": similarity,
                        "collection": collection.name
                    })
            
            # Sort by similarity and return top results
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:n_results]
            
        except Exception as e:
            logger.error(f"Error in test mode search: {e}")
            return []
    
    def _search_collection(self, collection_name: str, query: str, n_results: int = MAX_CANDIDATES) -> List[Dict]:
        """Search within a specific ChromaDB collection (creates new embedding - use _search_collection_with_embedding for efficiency)"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if self.test_mode:
                # In test mode, use simple text matching
                return self._test_mode_search(collection, query, n_results)
            
            query_embedding = self._create_embeddings([query])
            
            if not query_embedding:
                return []
            
            return self._search_collection_with_embedding(collection_name, query, query_embedding, n_results)
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def _search_collection_with_embedding(self, collection_name: str, query: str, query_embedding: List[List[float]], n_results: int = MAX_CANDIDATES) -> List[Dict]:
        """EFFICIENT: Search within a specific ChromaDB collection using pre-computed embedding"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if self.test_mode:
                # In test mode, use simple text matching
                return self._test_mode_search(collection, query, n_results)
            
            if not query_embedding:
                return []
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, collection.count()),
                include=["metadatas", "distances"]
            )
            
            candidates = []
            if results['metadatas'] and results['distances']:
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    # Convert distance to similarity (ChromaDB returns distances, we want similarity)
                    similarity = 1 - distance
                    
                    candidates.append({
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "source": metadata["source"],
                        "score": similarity,
                        "collection": collection_name
                    })
            
            return candidates
            
        except Exception as e:
            # 🛠️ ChromaDB Error Recovery
            error_msg = str(e).lower()
            if "hnsw segment reader" in error_msg or "nothing found on disk" in error_msg:
                logger.warning(f"ChromaDB corruption detected in {collection_name}, attempting recovery...")
                try:
                    # Try to recover by rebuilding this specific collection
                    if self._recover_collection(collection_name):
                        logger.info(f"Successfully recovered {collection_name}, retrying search...")
                        # Retry the search once after recovery
                        return self._search_collection_with_embedding(collection_name, query, query_embedding, n_results)
                except Exception as recovery_error:
                    logger.error(f"Collection recovery failed for {collection_name}: {recovery_error}")
            
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def _recover_collection(self, collection_name: str) -> bool:
        """🛠️ Simple recovery for corrupted ChromaDB collection"""
        try:
            # Find the source file for this collection
            file_mapping = {v: k for k, v in FILE_TO_COLLECTION.items()}
            source_filename = None
            
            for filename, coll_type in FILE_TO_COLLECTION.items():
                if f"faq_{coll_type}" == collection_name:
                    source_filename = filename
                    break
            
            if not source_filename:
                logger.error(f"No source file found for collection {collection_name}")
                return False
            
            # Delete corrupted collection
            try:
                self.chroma_client.delete_collection(collection_name)
                logger.info(f"Deleted corrupted collection: {collection_name}")
            except Exception:
                pass  # Collection might not exist
            
            # Rebuild from source file
            filepath = os.path.join(FAQ_DIR, source_filename)
            if os.path.exists(filepath):
                logger.info(f"Rebuilding {collection_name} from {source_filename}")
                faq_pairs = self._preprocess_faq_file(filepath)
                if faq_pairs:
                    self._update_collection(source_filename, faq_pairs)
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error recovering collection {collection_name}: {e}")
            return False
    
    def _search_all_collections(self, query: str) -> List[Dict]:
        """Search across all collections when no prime words detected (creates new embedding - use _search_all_collections_with_embedding for efficiency)"""
        all_candidates = []
        
        try:
            # Create embedding once for efficiency
            query_embedding = None
            if not self.test_mode:
                query_embedding = self._create_embeddings([query])
                if not query_embedding:
                    return []
            
            collections = self.chroma_client.list_collections()
            for collection in collections:
                candidates = self._search_collection_with_embedding(collection.name, query, query_embedding, MAX_CANDIDATES)
                all_candidates.extend(candidates)
        except Exception as e:
            logger.error(f"Error searching all collections: {e}")
        
        # Sort by similarity score and return top candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates[:MAX_CANDIDATES]
    
    def _search_all_collections_with_embedding(self, query: str, query_embedding: List[List[float]]) -> List[Dict]:
        """EFFICIENT: Search across all collections using pre-computed embedding"""
        all_candidates = []
        failed_collections = []
        
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                candidates = self._search_collection_with_embedding(collection.name, query, query_embedding, MAX_CANDIDATES)
                if candidates:
                    all_candidates.extend(candidates)
                else:
                    failed_collections.append(collection.name)
                    
        except Exception as e:
            logger.error(f"Error searching all collections: {e}")
        
        # 🛠️ Log any completely failed collections (optional recovery could go here)
        if failed_collections:
            logger.warning(f"Failed to search collections: {failed_collections}")
        
        # Sort by similarity score and return top candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates[:MAX_CANDIDATES]
    
    def _llm_semantic_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """🧠 LLM-powered semantic reranking for precise understanding (OPTIMIZED)"""
        if len(candidates) <= 1:
            return candidates
        
        # OPTIMIZATION: Only use LLM for ambiguous cases where top candidates are close
        top_scores = [c['score'] for c in candidates[:3]]
        score_variance = max(top_scores) - min(top_scores)
        
        # Skip LLM if there's a clear winner (saves API calls)
        if score_variance > 0.3:  # Clear differentiation exists
            return candidates
        
        # Only rerank if we have semantic ambiguity
        if not self._has_semantic_ambiguity(query, candidates[:5]):
            return candidates
        
        try:
            # Prepare concise prompt for LLM
            prompt = self._build_semantic_prompt(query, candidates[:8])  # Limit to top 8
            
            # Make LLM call
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for speed and cost efficiency
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Bengali banking expert. Rank FAQ matches by semantic relevance. Return only numbers 1,2,3... in order of best match."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=50,  # Very limited tokens needed
                temperature=0.1
            )
            
            # Parse LLM ranking
            ranking_text = response.choices[0].message.content.strip()
            new_order = self._parse_ranking(ranking_text, len(candidates[:8]))
            
            if new_order:
                # Apply LLM ranking to top candidates
                reranked_candidates = [candidates[i-1] for i in new_order if i <= len(candidates)]
                # Add remaining candidates
                remaining = [c for i, c in enumerate(candidates) if (i+1) not in new_order]
                return reranked_candidates + remaining
            
        except Exception as e:
            # Fail gracefully - return original ranking
            pass
        
        return candidates
    
    def _has_semantic_ambiguity(self, query: str, candidates: List[Dict]) -> bool:
        """Check if semantic reranking is needed"""
        query_lower = query.lower()
        
        # Check for semantic complexity indicators
        semantic_indicators = [
            'সুদ' in query_lower and any('ইন্টারেস্ট' in c['question'].lower() for c in candidates),
            'চার্জ' in query_lower and any('ফি' in c['question'].lower() for c in candidates),
            'খুলতে' in query_lower and any('ওপেন' in c['question'].lower() for c in candidates),
            len(set(c['collection'] for c in candidates)) > 1  # Cross-collection ambiguity
        ]
        
        return any(semantic_indicators)
    
    def _build_semantic_prompt(self, query: str, candidates: List[Dict]) -> str:
        """Build concise semantic prompt"""
        candidate_texts = []
        for i, candidate in enumerate(candidates, 1):
            candidate_texts.append(f"{i}. {candidate['question']}")
        
        return f"Query: {query}\n\nOptions:\n" + "\n".join(candidate_texts[:8])
    
    def _parse_ranking(self, ranking_text: str, max_candidates: int) -> List[int]:
        """Parse LLM ranking response"""
        import re
        numbers = re.findall(r'\d+', ranking_text)
        try:
            ranking = [int(n) for n in numbers if 1 <= int(n) <= max_candidates]
            return ranking[:max_candidates]  # Limit to available candidates
        except:
            return []
    
    def _find_best_match(self, query: str) -> Tuple[Optional[Dict], List[Dict]]:
        """🧠 ENHANCED: Find best match with advanced Bengali semantic understanding"""
        start_time = time.time()
        
        # 🇧🇩 NEW: Apply advanced Bengali semantic analysis
        semantic_analysis = self.bengali_semantic_engine.analyze_query(query)
        enhanced_query = semantic_analysis['enhanced_query']
        confidence_boost = semantic_analysis['confidence_boosts']['total']
        
        logger.info(f"🧠 Semantic analysis applied:")
        logger.info(f"   Original: {query}")
        logger.info(f"   Enhanced: {enhanced_query}")
        logger.info(f"   Question type: {semantic_analysis['morphology']['question_type']}")
        logger.info(f"   Banking context: {semantic_analysis['context']['banking_type']}")
        logger.info(f"   Confidence boost: {confidence_boost:.3f}")
        
        # Use enhanced query for search
        search_query = enhanced_query
        cleaned_query = self._clean_text(search_query)
        
        # Create embeddings for the enhanced query
        if not self.test_mode and self.client:
            start_embedding = time.time()
            query_embedding = self._create_embeddings([search_query])
            embedding_time = time.time() - start_embedding
            logger.debug(f"Embedding creation took {embedding_time:.3f}s")
        else:
            query_embedding = None
        
        # Detect prime words for initial routing using enhanced query
        detected_collections = self._detect_prime_words(search_query)
        
        # 🎯 ENHANCED: Apply semantic routing based on linguistic analysis
        if semantic_analysis['context']['banking_type'] == 'islamic':
            if 'yaqeen' not in detected_collections:
                detected_collections.append('yaqeen')
                logger.info("🕌 Added Islamic banking collection based on semantic analysis")
        
        # Add domain-specific collections based on context
        for domain in semantic_analysis['context']['domain_context']:
            if domain == 'business' and 'sme' not in detected_collections:
                detected_collections.append('sme')
                logger.info("🏢 Added SME collection based on business context")
        
        # Enhanced semantic mappings from analysis
        for category, matches in semantic_analysis['semantics'].items():
            if category == 'confidence_scores':
                continue
            if matches:  # If we have semantic matches in this category
                if category == 'account_related':
                    # Determine specific account type from semantic analysis
                    for match in matches:
                        if 'পেরোল' in match['replacement'] and 'payroll' not in detected_collections:
                            detected_collections.append('payroll')
                        elif 'এসএমই' in match['replacement'] and 'sme' not in detected_collections:
                            detected_collections.append('sme')
                        elif 'অঙ্গনা' in match['replacement'] and 'women' not in detected_collections:
                            detected_collections.append('women')
                        elif 'এনআরবি' in match['replacement'] and 'nrb' not in detected_collections:
                            detected_collections.append('nrb')
        
        logger.info(f"🎯 Enhanced routing detected collections: {detected_collections}")
        
        # Route to specific collections if prime words detected
        target_collections = []
        if detected_collections:
            intent_context = self._detect_banking_intent(search_query)
            target_collections = self._route_to_collections_enhanced(detected_collections, intent_context, semantic_analysis)
        
        # Initialize candidates list
        all_candidates = []
        search_all_needed = True
        
        # Search in targeted collections first
        if target_collections:
            logger.info(f"Searching in targeted collections: {target_collections}")
            targeted_candidates = []
            
            for collection in target_collections:
                collection_results = self._search_collection_with_embedding(
                    collection, search_query, query_embedding, n_results=5
                )
                if collection_results:
                    targeted_candidates.extend(collection_results)
            
            # 🧠 ENHANCED: Apply semantic confidence boost to targeted results
            for candidate in targeted_candidates:
                original_score = candidate['score']
                candidate['score'] = min(1.0, original_score + confidence_boost)
                candidate['semantic_boost_applied'] = confidence_boost
                logger.debug(f"Applied semantic boost: {original_score:.3f} → {candidate['score']:.3f}")
            
            all_candidates.extend(targeted_candidates)
            
            # SPEED OPTIMIZATION: Early exit for high-confidence matches
            # Check if we should also search all collections
            if all_candidates:
                best_targeted_score = max(c['score'] for c in all_candidates)
                
                # EARLY EXIT: If we have very high confidence match, skip searching all collections
                if best_targeted_score >= 0.95:
                    logger.info(f"High confidence match found ({best_targeted_score:.3f}), skipping all-collection search")
                    search_all_needed = False
                elif len(detected_collections) > 1:
                    logger.info("Multiple collections detected, will also search all collections")
                    search_all_needed = True
                elif best_targeted_score < 0.15:  # If best targeted match is very weak
                    logger.info(f"Best targeted match score {best_targeted_score:.3f} is very weak, will also search all collections")
                    search_all_needed = True
            elif len(detected_collections) > 1:
                logger.info("Multiple collections detected, will also search all collections")
                search_all_needed = True
        else:
            search_all_needed = True
        
        # Search all collections if needed - ENHANCED to get more candidates for semantic analysis
        if search_all_needed or not all_candidates:
            if not all_candidates:
                logger.info("No candidates from prime word routing, searching all collections")
            else:
                logger.info("Expanding search to all collections for better matches")
            
            # ENHANCED: Get more candidates with higher n_results for better semantic matching
            try:
                collections = self.chroma_client.list_collections()
                expanded_candidates = []
                
                for collection in collections:
                    # Get more results per collection for semantic analysis
                    if self.test_mode:
                        coll_candidates = self._test_mode_search(collection, search_query, n_results=5)
                    else:
                        coll_candidates = self._search_collection_with_embedding(
                            collection.name, search_query, query_embedding, n_results=5
                        )
                    
                    # Apply semantic boost to all candidates
                    for candidate in coll_candidates:
                        original_score = candidate['score']
                        candidate['score'] = min(1.0, original_score + confidence_boost)
                        candidate['semantic_boost_applied'] = confidence_boost
                    
                    expanded_candidates.extend(coll_candidates)
                
                # Merge with existing candidates, avoiding duplicates
                existing_questions = {c['question'] for c in all_candidates}
                for candidate in expanded_candidates:
                    if candidate['question'] not in existing_questions:
                        all_candidates.append(candidate)
                        
            except Exception as e:
                logger.error(f"Error in expanded search: {e}")
                # Fallback to original method
                all_collection_candidates = self._search_all_collections_with_embedding(search_query, query_embedding)
                existing_questions = {c['question'] for c in all_candidates}
                for candidate in all_collection_candidates:
                    if candidate['question'] not in existing_questions:
                        all_candidates.append(candidate)
            
            # Sort and keep more candidates for semantic analysis
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            all_candidates = all_candidates[:15]  # Keep even more for better semantic matching
        
        # Apply hybrid matching to enhance similarity scores
        logger.info(f"Enhancing {len(all_candidates)} candidates with hybrid matching")
        all_candidates = hybrid_enhance_candidates(
            candidates=all_candidates,
            query=search_query,  # Use enhanced query
            cleaned_query=cleaned_query,
            hybrid_matcher=self.hybrid_matcher
        )
        
        # 🚀 CRITICAL: Pre-boost exact semantic matches before LLM reranking
        all_candidates = self._boost_exact_semantic_matches(search_query, all_candidates)
        
        # ULTRA-ADVANCED: Cross-collection disambiguation and authority scoring
        intent_context = self._detect_banking_intent(search_query)
        all_candidates = self._apply_cross_collection_disambiguation(
            all_candidates, search_query.lower(), intent_context
        )
        
        # 🚀 NEW: LLM Semantic Reranking for improved semantic understanding
        logger.info(f"Applying LLM semantic reranking to {len(all_candidates)} candidates")
        all_candidates = self._llm_semantic_rerank(search_query, all_candidates)
        
        # 🇧🇩 ENHANCED: Apply Bengali-specific confidence adjustments
        all_candidates = self._apply_bengali_semantic_adjustments(all_candidates, semantic_analysis)
        
        # Dynamic threshold calculation based on cross-collection ambiguity
        dynamic_threshold = self._calculate_dynamic_threshold(all_candidates, intent_context)
        logger.info(f"Using dynamic confidence threshold: {dynamic_threshold:.3f}")
        
        # 🧠 ENHANCED: More intelligent threshold for semantic matches with Bengali context
        adjusted_threshold = self._calculate_adjusted_threshold(dynamic_threshold, all_candidates, semantic_analysis)
        
        # Final threshold decision
        if all_candidates and all_candidates[0]['score'] >= adjusted_threshold:
            logger.info(f"Best match: {all_candidates[0]['question'][:50]}... (score: {all_candidates[0]['score']:.3f}, threshold: {adjusted_threshold:.3f})")
            return all_candidates[0], all_candidates
        else:
            # No match found above threshold
            if all_candidates:
                logger.info(f"No match above threshold. Best candidate: {all_candidates[0]['question'][:50]}... (score: {all_candidates[0]['score']:.3f}, threshold: {adjusted_threshold:.3f})")
            else:
                logger.info("No candidates found")
            return None, all_candidates
    
    def _route_to_collections_enhanced(self, detected_collections: List[str], intent_context: Dict, semantic_analysis: Dict) -> List[str]:
        """🎯 Enhanced collection routing with semantic analysis"""
        enhanced_collections = detected_collections.copy()
        
        # Apply semantic-based routing enhancements
        banking_type = semantic_analysis['context']['banking_type']
        formality = semantic_analysis['context']['formality']
        
        # Islamic banking enhancement
        if banking_type == 'islamic' and 'yaqeen' not in enhanced_collections:
            enhanced_collections.append('yaqeen')
            logger.info("🕌 Enhanced routing: Added yaqeen for Islamic context")
        
        # Business context enhancement
        if 'business' in semantic_analysis['context']['domain_context'] and 'sme' not in enhanced_collections:
            enhanced_collections.append('sme')
            logger.info("🏢 Enhanced routing: Added sme for business context")
        
        # Formal query enhancement (privilege banking)
        if formality == 'high' and 'privilege' not in enhanced_collections:
            enhanced_collections.append('privilege')
            logger.info("👑 Enhanced routing: Added privilege for formal context")
        
        # 🛠️ FIXED: Convert collection types to actual collection names
        full_collection_names = []
        for collection_type in enhanced_collections:
            full_name = f"faq_{collection_type}"
            full_collection_names.append(full_name)
        
        return full_collection_names
    
    def _apply_bengali_semantic_adjustments(self, candidates: List[Dict], semantic_analysis: Dict) -> List[Dict]:
        """🇧🇩 Apply Bengali-specific semantic adjustments"""
        question_type = semantic_analysis['morphology']['question_type']
        banking_context = semantic_analysis['context']['banking_type']
        
        for candidate in candidates:
            adjustments = 0.0
            
            # Question type alignment bonus
            if question_type == 'quantity_quality' and any(word in candidate['question'] for word in ['কত', 'কেমন']):
                adjustments += 0.05
            elif question_type == 'yes_no_what' and any(word in candidate['question'] for word in ['কি', 'কী']):
                adjustments += 0.05
            
            # Banking context alignment
            if banking_context == 'islamic' and candidate['collection'] == 'faq_yaqeen':
                adjustments += 0.1
            elif banking_context == 'conventional' and candidate['collection'] in ['faq_retail', 'faq_nrb']:
                adjustments += 0.05
            
            # Apply adjustments
            if adjustments > 0:
                original_score = candidate['score']
                candidate['score'] = min(1.0, original_score + adjustments)
                candidate['bengali_semantic_adjustment'] = adjustments
                logger.debug(f"Bengali semantic adjustment: {original_score:.3f} → {candidate['score']:.3f} (+{adjustments:.3f})")
        
        # Re-sort after adjustments
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _calculate_adjusted_threshold(self, dynamic_threshold: float, candidates: List[Dict], semantic_analysis: Dict) -> float:
        """🎯 Calculate adjusted threshold with Bengali semantic context"""
        adjusted_threshold = dynamic_threshold
        
        # For semantic edge cases, be more lenient if we have good candidates
        if candidates:
            top_score = candidates[0]['score']
            question_type = semantic_analysis['morphology']['question_type']
            banking_type = semantic_analysis['context']['banking_type']
            
            # ENHANCED: Much more lenient semantic understanding
            if top_score >= 0.25:  # Lower threshold for semantic understanding
                
                # Find the best semantic match in all candidates, not just the top one
                best_semantic_candidate = None
                best_semantic_score = 0
                
                for candidate in candidates[:10]:  # Check top 10 candidates
                    semantic_score = self._calculate_semantic_equivalence(semantic_analysis['original'], candidate)
                    if semantic_score > best_semantic_score:
                        best_semantic_score = semantic_score
                        best_semantic_candidate = candidate
                
                if best_semantic_candidate and best_semantic_score > 0.6:
                    # Promote the semantically equivalent candidate to top
                    if best_semantic_candidate != candidates[0]:
                        candidates.remove(best_semantic_candidate)
                        candidates.insert(0, best_semantic_candidate)
                        logger.info(f"Promoted semantic match to top: {best_semantic_candidate['question'][:50]}...")
                    
                    adjusted_threshold = max(0.25, dynamic_threshold - 0.25)
                    logger.info(f"Strong semantic equivalence detected (score: {best_semantic_score:.2f}), lowering threshold to {adjusted_threshold:.3f}")
                elif top_score >= 0.70:
                    # Good match - be more lenient
                    adjusted_threshold = max(0.25, dynamic_threshold - 0.25)
                    logger.info(f"Good match detected (score: {top_score:.2f}), lowering threshold to {adjusted_threshold:.3f}")
                elif top_score >= 0.50:
                    # Decent match - slightly more lenient
                    adjusted_threshold = max(0.35, dynamic_threshold - 0.15)
                    logger.info(f"Decent match detected (score: {top_score:.2f}), slightly lowering threshold to {adjusted_threshold:.3f}")
                else:
                    # Weak match - minimal adjustment
                    adjusted_threshold = max(0.40, dynamic_threshold - 0.05)
                    logger.info(f"Weak match detected (score: {top_score:.2f}), minimal threshold reduction to {adjusted_threshold:.3f}")
                
                # 🇧🇩 ENHANCED: Bengali-specific threshold adjustments
                if question_type in ['quantity_quality', 'yes_no_what']:
                    adjusted_threshold = max(0.20, adjusted_threshold - 0.1)
                    logger.info(f"Bengali question type adjustment: threshold reduced to {adjusted_threshold:.3f}")
                
                if banking_type == 'islamic' and candidates[0]['collection'] == 'faq_yaqeen':
                    adjusted_threshold = max(0.15, adjusted_threshold - 0.15)
                    logger.info(f"Islamic banking context adjustment: threshold reduced to {adjusted_threshold:.3f}")
        
        return adjusted_threshold
    
    def _boost_exact_semantic_matches(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """🚀 CRITICAL: Boost candidates that are exact semantic matches"""
        query_lower = query.lower()
        
        # Define exact semantic equivalence patterns
        exact_patterns = [
            # Interest rate patterns - MOST CRITICAL
            {
                'query_pattern': r'(.+)(সুদের হার|সুদ হার)(.+)',
                'candidate_pattern': r'(.+)(ইন্টারসেট রেট|ইন্টারেস্ট রেট)(.+)',
                'boost': 0.7,  # Massive boost
                'description': 'Interest rate equivalence'
            },
            # Card fee patterns
            {
                'query_pattern': r'(.+)(কার্ডের ফি|কার্ড ফি)(.+)',
                'candidate_pattern': r'(.+)(কার্ড এর চার্জ|কার্ডের চার্জ)(.+)',
                'boost': 0.6,
                'description': 'Card fee equivalence'
            },
            # Account opening patterns
            {
                'query_pattern': r'(.+)(একাউন্ট খুলতে)(.+)',
                'candidate_pattern': r'(.+)(একাউন্ট ওপেন|অ্যাকাউন্ট ওপেন)(.+)',
                'boost': 0.5,
                'description': 'Account opening equivalence'
            }
        ]
        
        import re
        boosted_count = 0
        
        for candidate in candidates:
            candidate_question = candidate['question'].lower()
            original_score = candidate['score']
            
            for pattern in exact_patterns:
                query_match = re.search(pattern['query_pattern'], query_lower)
                candidate_match = re.search(pattern['candidate_pattern'], candidate_question)
                
                if query_match and candidate_match:
                    # Check if the context around the patterns is similar
                    query_context = (query_match.group(1) + query_match.group(3)).strip()
                    candidate_context = (candidate_match.group(1) + candidate_match.group(3)).strip()
                    
                    # Simple context similarity check
                    query_words = set(query_context.split())
                    candidate_words = set(candidate_context.split())
                    
                    if query_words and candidate_words:
                        context_overlap = len(query_words.intersection(candidate_words)) / len(query_words.union(candidate_words))
                        
                        # If context is reasonably similar, apply the boost
                        if context_overlap >= 0.4:  # At least 40% context overlap
                            boost_amount = pattern['boost'] * (0.5 + context_overlap * 0.5)  # Scale boost by context similarity
                            candidate['score'] = min(1.0, original_score + boost_amount)
                            candidate['semantic_boost'] = boost_amount
                            candidate['semantic_reason'] = pattern['description']
                            boosted_count += 1
                            
                            logger.info(f"🚀 Semantic boost applied: {pattern['description']} - "
                                      f"Score {original_score:.3f} → {candidate['score']:.3f} "
                                      f"(boost: +{boost_amount:.3f}, context: {context_overlap:.2f})")
                            break  # Only apply one boost per candidate
        
        if boosted_count > 0:
            # Re-sort candidates after boosting
            candidates.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"🚀 Applied semantic boosts to {boosted_count} candidates")
        
        return candidates
    
    def _calculate_semantic_equivalence(self, query: str, candidate: Dict) -> float:
        """Calculate semantic equivalence score for specific Bengali banking terms"""
        query_lower = query.lower()
        question_lower = candidate['question'].lower()
        
        # Define semantic equivalences with confidence scores
        equivalences = [
            # Interest rate equivalences - ENHANCED
            {
                'query_terms': ['সুদের হার', 'সুদ হার', 'সুদ'],
                'candidate_terms': ['ইন্টারেস্ট রেট', 'ইন্টারসেট রেট', 'ইন্টারেস্ট', 'ইন্টারসেট', 'রেট'],
                'score': 0.95
            },
            # Branch/Online equivalences
            {
                'query_terms': ['শাখায় যেতে', 'শাখায়', 'ব্রাঞ্চে'],
                'candidate_terms': ['অনলাইন', 'অনলাইনে', 'অনলাইন একাউন্ট'],
                'score': 0.90
            },
            # Account opening variations
            {
                'query_terms': ['একাউন্ট খুলতে'],
                'candidate_terms': ['একাউন্ট ওপেন', 'অ্যাকাউন্ট ওপেন', 'একাউন্ট খোলা'],
                'score': 0.85
            },
            # Payroll context - ENHANCED
            {
                'query_terms': ['পেরোল অ্যাকাউন্টের', 'পেরোল একাউন্টের', 'পেরোল'],
                'candidate_terms': ['পেরোল অ্যাকাউন্ট এর', 'পেরোল একাউন্ট এর', 'পেরোল', 'স্যালারি', 'বেতন'],
                'score': 0.85
            },
            # Card/Fee equivalences - ENHANCED
            {
                'query_terms': ['কার্ডের ফি', 'কার্ড ফি', 'ফি', 'চার্জ আছে', 'চার্জ'],
                'candidate_terms': ['কার্ড এর চার্জ', 'কার্ডের চার্জ', 'চার্জ', 'বাৎসরিক কোন চার্জ', 'বার্ষিক চার্জ'],
                'score': 0.95
            },
            # MTB Regular account equivalences - NEW
            {
                'query_terms': ['এমটিবি রেগুলার', 'রেগুলার অ্যাকাউন্ট', 'রেগুলার একাউন্ট'],
                'candidate_terms': ['এমটিবি এনআরবি', 'এনআরবি সেভিংস', 'নরব', 'এনআরবি'],
                'score': 0.80
            }
        ]
        
        max_score = 0
        
        for equiv in equivalences:
            query_match = any(term in query_lower for term in equiv['query_terms'])
            candidate_match = any(term in question_lower for term in equiv['candidate_terms'])
            
            if query_match and candidate_match:
                # Check for additional context overlap
                context_bonus = 0
                
                # If both mention the same domain (payroll, account, etc.)
                common_terms = ['পেরোল', 'একাউন্ট', 'অ্যাকাউন্ট']
                for term in common_terms:
                    if term in query_lower and term in question_lower:
                        context_bonus += 0.1
                
                final_score = min(1.0, equiv['score'] + context_bonus)
                max_score = max(max_score, final_score)
        
        return max_score
    
    def _detect_banking_intent(self, query: str) -> Dict:
        """Detect Islamic vs Conventional banking intent from query"""
        return self._detect_banking_intent_cached(query.lower())
    
    def _detect_banking_intent_cached(self, query_lower: str) -> Dict:
        """SPEED OPTIMIZED: Detect banking intent using pre-lowercased query"""
        islamic_indicators = [
            "ইয়াকিন", "ইসলামিক", "শরিয়া", "হালাল", "মুদারাবা", 
            "উজরা", "মুরাবাহা", "বাই", "অঘনিয়া", "আগানিয়া"
        ]
        
        conventional_indicators = [
            "এমটিবি", "রেগুলার", "ইন্সপায়ার", "সিনিয়র", "জুনিয়র",
            "এক্সট্রিম", "কেয়ার", "সঞ্চয়"
        ]
        
        # Neutral terms that could apply to both
        neutral_terms = ["লাখপতি", "কোটিপতি", "মিলিয়নিয়ার", "ডিপিএস"]
        
        islamic_score = sum(1 for indicator in islamic_indicators if indicator in query_lower)
        conventional_score = sum(1 for indicator in conventional_indicators if indicator in query_lower)
        neutral_score = sum(1 for term in neutral_terms if term in query_lower)
        
        # Normalize scores
        total_indicators = len(islamic_indicators) + len(conventional_indicators)
        islamic_intent = islamic_score / len(islamic_indicators) if islamic_indicators else 0
        conventional_intent = conventional_score / len(conventional_indicators) if conventional_indicators else 0
        
        # Determine dominant intent
        if islamic_score > conventional_score:
            dominant_intent = "islamic"
        elif conventional_score > islamic_score:
            dominant_intent = "conventional"
        else:
            dominant_intent = "ambiguous"
        
        return {
            'islamic_intent': islamic_intent,
            'conventional_intent': conventional_intent,
            'neutral_score': neutral_score,
            'dominant_intent': dominant_intent,
            'ambiguous': islamic_score == conventional_score and neutral_score > 0,
            'confidence': max(islamic_intent, conventional_intent)
        }
    
    def _apply_cross_collection_disambiguation(self, candidates: List[Dict], query_lower: str, intent_context: Dict) -> List[Dict]:
        """Apply cross-collection disambiguation with authority scoring"""
        if len(candidates) <= 1:
            return candidates
        
        # Group candidates by similarity (potential duplicates across collections)
        similarity_groups = self._find_similarity_groups(candidates)
        
        for group in similarity_groups:
            if len(group) > 1:  # Multiple similar candidates
                logger.info(f"Found {len(group)} similar candidates across collections, applying disambiguation")
                
                for candidate in group:
                    # Calculate authority score based on intent (using cached query_lower)
                    authority_multiplier = self._calculate_authority_score(
                        candidate, intent_context, query_lower
                    )
                    
                    # Apply authority multiplier to similarity score
                    original_score = candidate['score']
                    candidate['score'] = min(1.0, original_score * authority_multiplier)
                    candidate['authority_multiplier'] = authority_multiplier
                    candidate['original_score'] = original_score
                    
                    logger.debug(f"Collection {candidate['collection']}: "
                               f"Original={original_score:.3f}, Authority={authority_multiplier:.3f}, "
                               f"Final={candidate['score']:.3f}")
        
        # Re-sort based on adjusted scores
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _find_similarity_groups(self, candidates: List[Dict], similarity_threshold: float = 0.85) -> List[List[Dict]]:
        """Find groups of highly similar candidates across different collections"""
        from itertools import combinations
        
        groups = []
        processed = set()
        
        for i, candidate_a in enumerate(candidates):
            if i in processed:
                continue
                
            current_group = [candidate_a]
            processed.add(i)
            
            for j, candidate_b in enumerate(candidates[i+1:], i+1):
                if j in processed:
                    continue
                
                # Calculate similarity between questions
                similarity = self._calculate_question_similarity(
                    candidate_a['question'], candidate_b['question']
                )
                
                if similarity >= similarity_threshold:
                    current_group.append(candidate_b)
                    processed.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_question_similarity(self, question1: str, question2: str) -> float:
        """Calculate similarity between two questions using multiple methods"""
        # Simple approach using word overlap and n-grams
        words1 = set(question1.lower().split())
        words2 = set(question2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Length similarity penalty
        length_diff = abs(len(question1) - len(question2))
        max_length = max(len(question1), len(question2))
        length_penalty = 1 - (length_diff / max_length) if max_length > 0 else 0.0
        
        # Combined similarity
        combined_similarity = (jaccard * 0.7) + (length_penalty * 0.3)
        
        return combined_similarity
    
    def _calculate_authority_score(self, candidate: Dict, intent_context: Dict, query_lower: str) -> float:
        """SPEED OPTIMIZED: Calculate authority score using pre-lowercased query"""
        base_multiplier = 1.0
        collection = candidate['collection']
        
        # Intent-based authority scoring
        if intent_context['dominant_intent'] == 'islamic':
            if collection == 'faq_yaqeen':
                base_multiplier = 1.3  # Strong boost for Islamic banking collection
            else:
                base_multiplier = 0.7  # Penalty for non-Islamic collections
                
        elif intent_context['dominant_intent'] == 'conventional':
            if collection == 'faq_retail':
                base_multiplier = 1.2  # Boost for retail/conventional collection
            elif collection == 'faq_yaqeen':
                base_multiplier = 0.8  # Slight penalty for Islamic when conventional intended
                
        # Product name exact match bonus (using cached query_lower)
        question_lower = candidate['question'].lower()
        
        # Check for exact product name matches
        if "ইয়াকিন" in query_lower and "ইয়াকিন" in question_lower:
            if collection == 'faq_yaqeen':
                base_multiplier *= 1.2  # Extra boost for exact brand match
        
        if "এমটিবি" in query_lower and "এমটিবি" in question_lower:
            if collection == 'faq_retail':
                base_multiplier *= 1.2  # Extra boost for exact brand match
        
        # Collection specificity bonus
        collection_specificity = {
            'faq_yaqeen': 1.1,    # Highly specific
            'faq_retail': 1.0,    # General
            'faq_sme': 1.05,      # Moderately specific
            'faq_card': 1.05,     # Moderately specific
            'faq_women': 1.05,    # Moderately specific
            'faq_payroll': 1.05,  # Moderately specific
        }
        
        specificity_bonus = collection_specificity.get(collection, 1.0)
        
        # Ambiguity penalty - if intent is ambiguous, reduce confidence
        if intent_context['ambiguous']:
            ambiguity_penalty = 0.95
        else:
            ambiguity_penalty = 1.0
        
        final_multiplier = base_multiplier * specificity_bonus * ambiguity_penalty
        
        # Ensure multiplier is within reasonable bounds
        return max(0.5, min(1.5, final_multiplier))
    
    def _calculate_dynamic_threshold(self, candidates: List[Dict], intent_context: Dict) -> float:
        """Calculate dynamic confidence threshold based on cross-collection ambiguity"""
        base_threshold = CONFIDENCE_THRESHOLD
        
        if len(candidates) < 2:
            return base_threshold
        
        # Check for cross-collection competition
        top_collections = [c['collection'] for c in candidates[:3]]
        unique_collections = set(top_collections)
        
        if len(unique_collections) > 1:
            # Multiple collections in top candidates - potential ambiguity
            score_gap = candidates[0]['score'] - candidates[1]['score']
            
            # FIXED: Much more lenient threshold adjustments
            if score_gap < 0.02 and candidates[0]['collection'] != candidates[1]['collection']:
                # Only very high ambiguity should increase threshold slightly
                threshold_adjustment = 0.02
                logger.info(f"Very high cross-collection ambiguity detected (gap={score_gap:.3f}), "
                          f"increasing threshold by {threshold_adjustment}")
            elif score_gap < 0.05:
                # Moderate ambiguity - minimal adjustment
                threshold_adjustment = 0.01
                logger.info(f"Moderate cross-collection ambiguity detected (gap={score_gap:.3f}), "
                          f"increasing threshold by {threshold_adjustment}")
            else:
                # Clear winner - reduce threshold
                threshold_adjustment = -0.05
                logger.info(f"Clear winner detected (gap={score_gap:.3f}), reducing threshold")
        else:
            # Single collection dominance - can be more lenient
            threshold_adjustment = -0.05
            logger.info("Single collection dominance, reducing threshold")
        
        # FIXED: Remove intent ambiguity penalty - it was too aggressive
        # Intent confidence is already handled in authority scoring
        
        dynamic_threshold = base_threshold + threshold_adjustment
        
        # FIXED: Much more reasonable bounds - don't go above 0.85
        return max(0.4, min(0.85, dynamic_threshold))
    
    def initialize(self) -> bool:
        """
        Initializes the system by loading and preprocessing all FAQ data.
        :return: A boolean indicating if initialization was successful.
        """
        try:
            logger.info("Initializing Bengali FAQ Service...")
            
            # 1. Check for updates in FAQ files
            needs_update, files_to_process = self._check_for_updates()
            
            if needs_update:
                logger.info("FAQ data has changed. Starting update process...")
                
                # Process all files that are new or have been modified
                for filename in files_to_process:
                    filepath = os.path.join(FAQ_DIR, filename)
                    logger.info(f"Processing file: {filepath}")
                    
                    faq_pairs = self._preprocess_faq_file(filepath)
                    self._update_collection(filename, faq_pairs)
                
                # Save the updated hashes to cache
                self._save_file_hashes()
                logger.info("Update process finished. File hashes are cached.")
            else:
                logger.info("No changes detected in FAQ files. System is up-to-date.")
            
            self.initialized = True
            logger.info("✅ FAQ Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAQ Service: {e}")
            logger.error(traceback.format_exc())
            self.initialized = False
            return False
    
    async def answer_query_async(self, query: str, debug: bool = False) -> Dict:
        """
        Asynchronously finds the best answer for a given query.
        This is a wrapper for the synchronous answer_query method.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.answer_query, query, debug
        )

    def answer_query(self, query: str, debug: bool = False) -> Dict:
        """
        Finds the best answer for a given query. This is the main entry point for querying.
        
        :param query: The user's question.
        :param debug: If True, returns detailed debug information.
        :return: A dictionary with the result.
        """
        try:
            # Input validation
            if query is None:
                return {
                    "found": False,
                    "query": None,
                    "error": "Query cannot be None",
                    "message": "দয়া করে একটি প্রশ্ন লিখুন।"
                }
            
            if not isinstance(query, str):
                return {
                    "found": False,
                    "query": str(query),
                    "error": f"Query must be a string, got {type(query).__name__}",
                    "message": "দয়া করে একটি বৈধ প্রশ্ন লিখুন।"
                }
            
            # Strip and check if empty
            query = query.strip()
            if not query:
                return {
                    "found": False,
                    "query": "",
                    "message": "দয়া করে একটি প্রশ্ন লিখুন।",
                    "confidence": 0.0
                }
            
            # Clean the incoming query to match the cleaning of stored questions
            cleaned_query = self._clean_text(query)
            
            # Find the best match using the cleaned query
            match, candidates = self._find_best_match(cleaned_query)
            
            if match:  # Trust _find_best_match's decision on threshold
                response = {
                    "found": True,
                    "query": query, # Return original query
                    "matched_question": match["question"],
                    "answer": match["answer"],
                    "source": match["source"],
                    "collection": match.get("collection", "unknown"),
                    "confidence": match["score"]
                }
            else:
                # Get the score of the best-rejected candidate for reporting
                best_score = max([c.get("score", 0) for c in candidates]) if candidates else 0.0
                
                response = {
                    "found": False,
                    "query": query,
                    "message": "দুঃখিত, আমি আপনার প্রশ্নের উত্তর খুঁজে পাইনি।",
                    "confidence": best_score
                }
            
            if debug:
                response["candidates"] = candidates
                response["threshold"] = CONFIDENCE_THRESHOLD
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering query '{query}': {e}")
            logger.error(traceback.format_exc())
            return {"found": False, "query": query, "error": str(e)}
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive statistics about the running system"""
        try:
            stats = {
                "test_mode": self.test_mode,
                "initialized": self.initialized,
                "version": "2.0.0",
                "service_status": "healthy" if self.initialized else "not_initialized"
            }
            
            # Collection statistics
            collections = self.chroma_client.list_collections()
            stats["total_collections"] = len(collections)
            stats["collections"] = {}
            
            total_entries = 0
            for collection in collections:
                try:
                    coll = self.chroma_client.get_collection(collection.name)
                    count = coll.count()
                    total_entries += count
                    
                    stats["collections"][collection.name] = {
                        "count": count,
                        "metadata": collection.metadata or {},
                        "status": "healthy"
                    }
                except Exception as e:
                    stats["collections"][collection.name] = {
                        "count": 0,
                        "metadata": {},
                        "status": "error",
                        "error": str(e)
                    }
            
            stats["total_entries"] = total_entries
            
            # Performance statistics - check if performance_monitor exists
            stats["performance"] = {
                "avg_query_time": 0.0,
                "cache_hit_rate": 0.0,
                "total_queries": 0
            }
            
            # Cache statistics - check if embedding_cache exists
            stats["cache"] = {
                "size": 0,
                "max_size": 1000,
                "ttl_hours": 24
            }
            
            # Configuration
            stats["config"] = {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "max_candidates": MAX_CANDIDATES,
                "faq_dir": FAQ_DIR,
                "embedding_model": EMBEDDING_MODEL
            }
            
            # Health status
            healthy_collections = sum(1 for c in stats["collections"].values() if c["status"] == "healthy")
            error_collections = len(stats["collections"]) - healthy_collections
            
            stats["health"] = {
                "overall_status": "healthy" if error_collections == 0 else "degraded",
                "healthy_collections": healthy_collections,
                "error_collections": error_collections,
                "service_uptime": "available" if self.initialized else "not_available"
            }
            
            # Memory and system info
            import sys
            
            stats["system"] = {
                "python_version": sys.version.split()[0],
                "memory_usage_mb": 0,
                "cpu_percent": 0,
                "threads": 1
            }
            
            # Try to get detailed system info if psutil is available
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                
                stats["system"].update({
                    "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads()
                })
            except ImportError:
                # psutil not available, use basic info
                pass
            except Exception:
                # Other errors getting system info
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "error": str(e),
                "test_mode": getattr(self, 'test_mode', False),
                "initialized": getattr(self, 'initialized', False),
                "service_status": "error"
            }

    def health_check(self) -> Dict:
        """🛠️ Simple health check for ChromaDB collections"""
        try:
            if not self.initialized:
                return {"status": "error", "message": "Service not initialized"}
            
            collections = self.chroma_client.list_collections()
            healthy_collections = []
            corrupted_collections = []
            
            for collection in collections:
                try:
                    coll = self.chroma_client.get_collection(collection.name)
                    count = coll.count()
                    
                    # 🛠️ REMOVED: No test query to avoid triggering ONNX downloads
                    # Corruption will be detected during actual searches only
                    
                    healthy_collections.append({
                        "name": collection.name,
                        "count": count
                    })
                    
                except Exception as e:
                    corrupted_collections.append({
                        "name": collection.name,
                        "error": str(e)
                    })
            
            status = "healthy" if not corrupted_collections else "degraded"
            
            return {
                "status": status,
                "healthy_collections": len(healthy_collections),
                "corrupted_collections": len(corrupted_collections),
                "details": {
                    "healthy": healthy_collections,
                    "corrupted": corrupted_collections
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {e}"}

    def _route_to_collections(self, query: str) -> List[str]:
        """🎯 Enhanced routing to find the most relevant collections first"""
        query_lower = query.lower()
        collections = []
        
        # ENHANCED: More specific routing patterns with priority
        routing_patterns = {
            # Islamic Banking - High priority patterns
            'yaqeen': {
                'patterns': ['ইয়াকিন', 'yaqeen', 'ইসলামিক', 'islamic', 'শরিয়া', 'shariah', 'প্রফিট রেট', 'profit rate'],
                'priority': 10,
                'exclusions': ['পেরোল', 'payroll']  # Exclude when it's Islamic payroll
            },
            
            # Payroll Banking - High priority 
            'payroll': {
                'patterns': ['পেরোল', 'payroll', 'বেতন', 'salary', 'স্যালারি'],
                'priority': 10,
                'inclusions': ['একাউন্ট', 'account', 'সুবিধা', 'benefit']  # Must include these for payroll
            },
            
            # Women Banking - High priority
            'women': {
                'patterns': ['অঙ্গনা', 'angona', 'মহিলা', 'women', 'নারী', 'lady'],
                'priority': 10,
                'context_boost': ['বিশেষ', 'special', 'একাউন্ট', 'account']
            },
            
            # SME Banking - Medium priority
            'sme': {
                'patterns': ['এসএমই', 'sme', 'ব্যবসা', 'business', 'উদ্যোক্তা', 'entrepreneur', 'ব্যবসায়ী'],
                'priority': 8,
                'exclusions': ['মহিলা', 'women', 'অঙ্গনা']  # Don't confuse with women's business
            },
            
            # NRB Banking 
            'nrb': {
                'patterns': ['এনআরবি', 'nrb', 'প্রবাসী', 'expatriate', 'রেমিটেন্স', 'remittance'],
                'priority': 9
            },
            
            # Agent Banking
            'agent': {
                'patterns': ['এজেন্ট', 'agent'],
                'priority': 9
            },
            
            # Privilege Banking  
            'privilege': {
                'patterns': ['প্রিভিলেজ', 'privilege'],
                'priority': 9
            },
            
            # Card Services
            'card': {
                'patterns': ['কার্ড', 'card', 'ডেবিট', 'debit', 'ক্রেডিট', 'credit'],
                'priority': 7
            },
            
            # Retail Banking - Lower priority (catch-all)
            'retail': {
                'patterns': ['রেগুলার', 'regular', 'সাধারণ', 'general', 'সেভিংস', 'savings'],
                'priority': 5
            }
        }
        
        # Calculate routing scores
        routing_scores = {}
        
        for collection, config in routing_patterns.items():
            score = 0
            patterns = config['patterns']
            priority = config.get('priority', 5)
            
            # Check main patterns
            pattern_matches = sum(1 for pattern in patterns if pattern in query_lower)
            if pattern_matches > 0:
                score += pattern_matches * priority
                
                # Check inclusions (must have these for the collection to be valid)
                inclusions = config.get('inclusions', [])
                if inclusions:
                    inclusion_matches = sum(1 for inc in inclusions if inc in query_lower)
                    if inclusion_matches == 0:
                        score = 0  # Invalidate if required inclusions not found
                
                # Check exclusions (can't have these)
                exclusions = config.get('exclusions', [])
                if exclusions:
                    exclusion_matches = sum(1 for exc in exclusions if exc in query_lower)
                    if exclusion_matches > 0:
                        score = 0  # Invalidate if exclusions found
                
                # Apply context boost
                context_boost = config.get('context_boost', [])
                if context_boost:
                    boost_matches = sum(1 for boost in context_boost if boost in query_lower)
                    score += boost_matches * 2
                
                routing_scores[collection] = score
        
        # Sort by score and return top collections
        sorted_collections = sorted(routing_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top scoring collections (score > 0)
        collections = [coll for coll, score in sorted_collections if score > 0]
        
        # ENHANCED: Handle Islamic Payroll special case
        if 'ইসলামিক' in query_lower and 'পেরোল' in query_lower:
            # For Islamic Payroll, prefer Yaqeen over Payroll
            if 'yaqeen' in collections and 'payroll' in collections:
                collections = ['yaqeen'] + [c for c in collections if c != 'yaqeen']
        
        # ENHANCED: Handle Women's Business special case  
        if any(pattern in query_lower for pattern in ['মহিলা', 'women', 'নারী']) and any(pattern in query_lower for pattern in ['ব্যবসা', 'business']):
            # For women's business, prefer women over SME
            if 'women' in collections and 'sme' in collections:
                collections = ['women'] + [c for c in collections if c != 'women']
        
        logger.info(f"Detected collections for routing: {collections}")
        return collections[:3]  # Return top 3 collections

# Global service instance - auto-initialize on import
faq_service = BengaliFAQService()

# Auto-initialize the service when module is imported
logger.info("Auto-initializing Bengali FAQ Service...")
initialization_success = faq_service.initialize()

if initialization_success:
    logger.info("✅ Bengali FAQ Service initialized successfully!")
else:
    logger.error("❌ Bengali FAQ Service initialization failed!")

# Export the service instance
__all__ = ['faq_service'] 