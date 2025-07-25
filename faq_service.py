import os
import json
import hashlib
import asyncio
import logging
import traceback
import glob
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Import hybrid matcher
from hybrid_matcher import HybridMatcher, hybrid_enhance_candidates

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
EMBEDDING_MODEL = config['models']['embedding_model']
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

class BengaliFAQService:
    """Core Bengali FAQ Service using ChromaDB with file-as-cluster routing"""
    
    def __init__(self):
        # Handle missing API key gracefully
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.test_mode = False
        else:
            self.client = None
            self.test_mode = True
            logger.warning("OPENAI_API_KEY not found. Running in test mode (no embeddings).")
        
        self.file_hashes = {}
        self.initialized = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Initialize hybrid matcher
        self.hybrid_matcher = HybridMatcher(config)
        
        # Load file hashes
        self._load_file_hashes()
    
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
        """Check if FAQ files have been modified since last run"""
        files_to_process = set()
        
        discovered_files = self._discover_faq_files()
        
        for filename in discovered_files:
            filepath = os.path.join(FAQ_DIR, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Warning: File {filepath} does not exist.")
                continue
                
            current_hash = self._calculate_file_hash(filepath)
            if current_hash:
                if filename not in self.file_hashes or self.file_hashes[filename] != current_hash:
                    files_to_process.add(filename)
                    self.file_hashes[filename] = current_hash
        
        return len(files_to_process) > 0, files_to_process
    
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
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using configured embedding model"""
        if not texts:
            return []
        
        if self.test_mode:
            # In test mode, return dummy embeddings
            logger.info(f"Test mode: Creating {len(texts)} dummy embeddings")
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSIONS
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []
    
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
        """Use LLM to semantically rerank candidates for better semantic understanding"""
        if not candidates or self.test_mode or not self.client:
            return candidates
        
        # ENHANCED: Always try LLM reranking for better semantic matches, even with single candidates
        # This helps with edge cases where embeddings miss semantic similarity
        
        try:
            # Prepare context with query + top candidates (limit to top 8 for context efficiency)
            top_candidates = candidates[:8]
            context = f"Query: {query}\n\nFAQ Options:\n"
            
            for i, candidate in enumerate(top_candidates):
                context += f"{i+1}. Q: {candidate['question']}\n   A: {candidate['answer'][:200]}...\n   Source: {candidate['source']}\n\n"
            
            prompt = f"""You are a Bengali banking FAQ expert. Given this query and FAQ options, rank them by SEMANTIC SIMILARITY, not exact word matching.

CRITICAL Bengali Banking Equivalencies:
- সুদের হার = ইন্টারেস্ট রেট = ইন্টারসেট রেট = রেট (ALL mean interest rate)
- শাখায় যেতে = ব্রাঞ্চে যেতে ≠ অনলাইনে (branch visit vs online)
- একাউন্ট = অ্যাকাউন্ট = হিসাব (account variations)
- পেরোল = স্যালারি = বেতন (payroll/salary)
- চার্জ = ফি = খরচ (fees/charges)
- কার্ডের ফি = কার্ড এর চার্জ (card fees)

PRIORITIZE SEMANTIC MEANING OVER EXACT WORDS.
Example: "সুদের হার কত?" should match "ইন্টারসেট রেট কত?" very highly.

{context}

Return ONLY the numbers (1-{len(top_candidates)}) in order of BEST SEMANTIC MATCH (meaning similarity), separated by commas.
Example: 3,1,5,2,4"""
            
            # Call LLM for semantic ranking with fallback models
            models_to_try = [
                config['models'].get('core_model', 'gpt-4.1'),
                'gpt-4.1-nano'
            ]
            
            response = None
            for model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    logger.debug(f"Successfully used model: {model}")
                    break
                except Exception as model_error:
                    logger.debug(f"Model {model} failed: {model_error}")
                    continue
            
            if not response:
                logger.warning("All LLM models failed, falling back to original ranking")
                return candidates
            
            ranking_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM ranking response: {ranking_text}")
            
            # Parse the ranking
            try:
                rankings = [int(x.strip()) - 1 for x in ranking_text.split(',') if x.strip().isdigit()]
                
                # Reorder candidates based on LLM ranking
                reranked = []
                used_indices = set()
                
                for rank_idx in rankings:
                    if 0 <= rank_idx < len(top_candidates) and rank_idx not in used_indices:
                        reranked.append(top_candidates[rank_idx])
                        used_indices.add(rank_idx)
                
                # Add any remaining candidates
                for i, candidate in enumerate(top_candidates):
                    if i not in used_indices:
                        reranked.append(candidate)
                
                # Add remaining original candidates that weren't in top 8
                if len(candidates) > 8:
                    reranked.extend(candidates[8:])
                
                logger.info(f"LLM reranked {len(top_candidates)} candidates")
                return reranked
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing LLM ranking '{ranking_text}': {e}")
                return candidates
                
        except Exception as e:
            logger.error(f"Error in LLM semantic reranking: {e}")
            return candidates
    
    def _find_best_match(self, query: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Find the best match using two-stage routing with hybrid matching and LLM semantic reranking"""
        # SPEED OPTIMIZATION: Cache repeated calculations within same query
        cleaned_query = self._clean_text(query)
        
        # EFFICIENCY: Create query embedding ONCE and reuse
        query_embedding = None
        if not self.test_mode:
            query_embedding = self._create_embeddings([query])
            if not query_embedding:
                logger.error("Failed to create query embedding")
                return None, []
        
        # Stage 1: Intent detection (cache query_lower for reuse)
        query_lower = query.lower()
        intent_context = self._detect_banking_intent_cached(query_lower)
        
        # Stage 2: Prime word detection (reuse query_lower)
        detected_collections = self._detect_prime_words_cached(query_lower)
        
        all_candidates = []
        search_all_needed = False
        
        if detected_collections:
            logger.info(f"Detected collections for routing: {detected_collections}")
            # Search in detected collections with cached embedding
            for collection_type in detected_collections:
                collection_name = f"faq_{collection_type}"
                candidates = self._search_collection_with_embedding(collection_name, query, query_embedding)
                all_candidates.extend(candidates)
            
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
                elif best_targeted_score < 0.7:  # If best targeted match is weak
                    logger.info(f"Best targeted match score {best_targeted_score:.3f} is weak, will also search all collections")
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
                        coll_candidates = self._test_mode_search(collection, query, n_results=5)
                    else:
                        coll_candidates = self._search_collection_with_embedding(
                            collection.name, query, query_embedding, n_results=5
                        )
                    expanded_candidates.extend(coll_candidates)
                
                # Merge with existing candidates, avoiding duplicates
                existing_questions = {c['question'] for c in all_candidates}
                for candidate in expanded_candidates:
                    if candidate['question'] not in existing_questions:
                        all_candidates.append(candidate)
                        
            except Exception as e:
                logger.error(f"Error in expanded search: {e}")
                # Fallback to original method
                all_collection_candidates = self._search_all_collections_with_embedding(query, query_embedding)
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
            query=query,
            cleaned_query=cleaned_query,
            hybrid_matcher=self.hybrid_matcher
        )
        
        # 🚀 CRITICAL: Pre-boost exact semantic matches before LLM reranking
        all_candidates = self._boost_exact_semantic_matches(query, all_candidates)
        
        # ULTRA-ADVANCED: Cross-collection disambiguation and authority scoring
        all_candidates = self._apply_cross_collection_disambiguation(
            all_candidates, query_lower, intent_context
        )
        
        # 🚀 NEW: LLM Semantic Reranking for improved semantic understanding
        logger.info(f"Applying LLM semantic reranking to {len(all_candidates)} candidates")
        all_candidates = self._llm_semantic_rerank(query, all_candidates)
        
        # Dynamic threshold calculation based on cross-collection ambiguity
        dynamic_threshold = self._calculate_dynamic_threshold(all_candidates, intent_context)
        logger.info(f"Using dynamic confidence threshold: {dynamic_threshold:.3f}")
        
        # ENHANCED: More intelligent threshold for semantic matches
        adjusted_threshold = dynamic_threshold
        
        # For semantic edge cases, be more lenient if we have good candidates
        if all_candidates:
            top_score = all_candidates[0]['score']
            
            # ENHANCED: Semantic equivalence detection for better matching
            if top_score >= 0.30:  # Even lower threshold for semantic understanding
                
                # Find the best semantic match in all candidates, not just the top one
                best_semantic_candidate = None
                best_semantic_score = 0
                
                for candidate in all_candidates[:10]:  # Check top 10 candidates
                    semantic_score = self._calculate_semantic_equivalence(query, candidate)
                    if semantic_score > best_semantic_score:
                        best_semantic_score = semantic_score
                        best_semantic_candidate = candidate
                
                if best_semantic_candidate and best_semantic_score > 0.7:
                    # Promote the semantically equivalent candidate to top
                    if best_semantic_candidate != all_candidates[0]:
                        all_candidates.remove(best_semantic_candidate)
                        all_candidates.insert(0, best_semantic_candidate)
                        logger.info(f"Promoted semantic match to top: {best_semantic_candidate['question'][:50]}...")
                    
                    adjusted_threshold = max(0.30, dynamic_threshold - 0.20)
                    logger.info(f"Strong semantic equivalence detected (score: {best_semantic_score:.2f}), lowering threshold to {adjusted_threshold:.3f}")
                else:
                    adjusted_threshold = max(0.35, dynamic_threshold - 0.15)
        
        # Check if best match meets adjusted confidence threshold
        if all_candidates and all_candidates[0]['score'] >= adjusted_threshold:
            logger.info(f"Best match: {all_candidates[0]['question'][:50]}... (score: {all_candidates[0]['score']:.3f}, threshold: {adjusted_threshold:.3f})")
            return all_candidates[0], all_candidates
        
        return None, all_candidates
    
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
            # Card/Fee equivalences
            {
                'query_terms': ['কার্ডের ফি', 'কার্ড ফি', 'ফি'],
                'candidate_terms': ['কার্ড এর চার্জ', 'কার্ডের চার্জ', 'চার্জ'],
                'score': 0.90
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
            
            # If scores are very close and from different collections
            if score_gap < 0.05 and candidates[0]['collection'] != candidates[1]['collection']:
                # High ambiguity - require higher confidence
                threshold_adjustment = 0.05
                logger.info(f"High cross-collection ambiguity detected (gap={score_gap:.3f}), "
                          f"increasing threshold by {threshold_adjustment}")
            elif score_gap < 0.1:
                # Moderate ambiguity
                threshold_adjustment = 0.02
                logger.info(f"Moderate cross-collection ambiguity detected (gap={score_gap:.3f}), "
                          f"increasing threshold by {threshold_adjustment}")
            else:
                threshold_adjustment = 0.0
        else:
            # Single collection dominance - can be more lenient
            threshold_adjustment = -0.02
            logger.info("Single collection dominance, slightly reducing threshold")
        
        # Intent confidence factor
        if intent_context['ambiguous']:
            threshold_adjustment += 0.03  # Extra caution for ambiguous intent
        
        dynamic_threshold = base_threshold + threshold_adjustment
        
        # Ensure threshold stays within reasonable bounds
        return max(0.4, min(0.95, dynamic_threshold))
    
    def initialize(self) -> bool:
        """Initialize the system by loading and preprocessing all FAQ data"""
        try:
            logger.info("Initializing Bengali FAQ Service...")
            
            if not os.path.exists(FAQ_DIR):
                logger.error(f"Error: FAQ directory '{FAQ_DIR}' does not exist.")
                return False
            
            discovered_files = self._discover_faq_files()
            if not discovered_files:
                logger.error("No .txt files found in FAQ directory.")
                return False
            
            # Check for updates
            updates_needed, files_to_process = self._check_for_updates()
            
            if not updates_needed:
                # Check if collections exist AND have data (WITHOUT triggering ONNX downloads)
                try:
                    collections = self.chroma_client.list_collections()
                    if collections:
                        total_entries = 0
                        
                        for collection in collections:
                            try:
                                coll = self.chroma_client.get_collection(collection.name)
                                count = coll.count()
                                total_entries += count
                                
                                # 🛠️ REMOVED: No more test queries that trigger ONNX downloads!
                                # The corruption detection will happen during actual searches only
                                    
                            except Exception as e:
                                logger.warning(f"Error checking collection {collection.name}: {e}")
                                # Mark this collection for rebuilding if there's an issue
                                for filename, coll_type in FILE_TO_COLLECTION.items():
                                    if f"faq_{coll_type}" == collection.name:
                                        files_to_process.add(filename)
                                        break
                        
                        if total_entries > 0:
                            logger.info(f"No updates needed. Using existing ChromaDB collections with {total_entries} total entries.")
                            self.initialized = True
                            return True
                        else:
                            logger.info("Found existing collections but they are empty. Reprocessing all files.")
                            files_to_process = set(discovered_files)
                    else:
                        logger.info("No existing collections found, processing all files.")
                        files_to_process = set(discovered_files)
                        
                except Exception as e:
                    logger.warning(f"Error checking existing collections: {e}, processing all files.")
                    files_to_process = set(discovered_files)
            
            if not files_to_process:
                files_to_process = set(discovered_files)
            
            logger.info(f"Processing {len(files_to_process)} files...")
            
            # Process each file and update its collection
            for filename in files_to_process:
                filepath = os.path.join(FAQ_DIR, filename)
                if os.path.exists(filepath):
                    logger.info(f"Processing file: {filepath}")
                    faq_pairs = self._preprocess_faq_file(filepath)
                    if faq_pairs:
                        self._update_collection(filename, faq_pairs)
                    else:
                        logger.warning(f"No FAQ pairs extracted from {filepath}")
            
            # Save file hashes
            self._save_file_hashes()
            
            # Verify collections were created
            collections = self.chroma_client.list_collections()
            logger.info(f"Initialization complete. Created {len(collections)} collections.")
            self.initialized = True
            return len(collections) > 0
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def answer_query_async(self, query: str, debug: bool = False) -> Dict:
        """Answer a user query using the optimized routing system with hybrid matching"""
        if not self.initialized:
            logger.error("FAQ Service not initialized. Call initialize() first.")
            return {"found": False, "message": "System not initialized"}
        
        try:
            # Clean the query
            cleaned_query = self._clean_text(query)
            
            # Check if query is in Bengali
            has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
            
            # Find best match using two-stage routing with hybrid matching
            logger.info(f"Processing query: {cleaned_query}")
            best_match, all_candidates = self._find_best_match(query)  # Pass original query, not cleaned
            
            result = {"found": False, "confidence": 0.0}
            
            if best_match:
                result.update({
                    "found": True,
                    "matched_question": best_match["question"],
                    "answer": best_match["answer"],
                    "source": best_match["source"],
                    "confidence": best_match["score"],
                    "collection": best_match["collection"]
                })
                logger.info(f"Found match with confidence {best_match['score']:.3f} from {best_match['collection']}")
            else:
                # Return fallback message in appropriate language
                if has_bengali:
                    result["message"] = "দুঃখিত, আমি আপনার প্রশ্নের উত্তর খুঁজে পাইনি। অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লিখুন।"
                else:
                    result["message"] = "Sorry, I couldn't find an answer to your question. Please rephrase your question."
                
                if all_candidates:
                    result["confidence"] = all_candidates[0]["score"]
                    logger.info(f"Best candidate score {all_candidates[0]['score']:.3f} below threshold {CONFIDENCE_THRESHOLD}")
            
            # Add debug info if requested
            if debug:
                result["candidates"] = all_candidates[:5]  # Top 5 candidates
                result["detected_collections"] = self._detect_prime_words(cleaned_query)
                result["threshold"] = CONFIDENCE_THRESHOLD
                
                # Add hybrid matching details for top candidate
                if all_candidates and "match_details" in all_candidates[0]:
                    result["match_details"] = all_candidates[0]["match_details"]
            
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
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            collections = self.chroma_client.list_collections()
            stats = {
                "total_collections": len(collections),
                "collections": {},
                "initialized": self.initialized,
                "test_mode": self.test_mode
            }
            
            for collection in collections:
                coll = self.chroma_client.get_collection(collection.name)
                stats["collections"][collection.name] = {
                    "count": coll.count(),
                    "metadata": collection.metadata
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

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
