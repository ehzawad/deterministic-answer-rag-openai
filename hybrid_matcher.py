#!/usr/bin/env python3
"""
Hybrid Matching Module for Bengali FAQ System

This module provides enhanced matching capabilities by combining:
1. Exact string matching
2. N-gram and keyword similarity with boosting for important terms
3. Embedding semantic similarity with weighted combination

This approach balances exact matching with real-world flexibility for
semantics, providing reasonable scores for contextually similar questions.
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import Counter
import numpy as np

# Configure logging
logger = logging.getLogger('BengaliFAQ-HybridMatcher')

class HybridMatcher:
    def __init__(self, config=None):
        """Initialize the hybrid matcher with configuration"""
        self.config = config or {}
        
        # Default weights for hybrid scoring
        self.weights = {
            'exact_match': 1.0,     # Weight for exact string match
            'cleaned_match': 0.95,  # Weight for cleaned text exact match
            'ngram_match': 0.4,     # Weight for n-gram matching score (increased)
            'keyword_match': 0.7,   # Weight for keyword matching score (increased)
            'embedding': 0.8,       # Weight for embedding similarity (increased)
            'phrase_match': 0.6,    # Weight for phrase matching
            'boost_factor': 0.15    # Boost factor for strong non-exact matches
        }
        
        # Update weights from config if provided
        if 'matcher_weights' in self.config:
            self.weights.update(self.config.get('matcher_weights', {}))
        
        # Define important Islamic banking phrases in Bengali
        self.banking_phrases = {
            'ইসলামিক ব্যাংকিং': 3.0,
            'ইসলামি ব্যাংক': 3.0,
            'শরীয়া ব্যাংকিং': 3.0,
            'শরিয়া আইন': 3.0,
            'মুদারাবা': 2.5,
            'মুশারাকা': 2.5,
            'ইয়াকিন সেভিংস': 2.5,
            'সেভিংস অ্যাকাউন্ট': 2.0,
            'ডেবিট কার্ড': 2.0,
            'কার্রেন্ট অ্যাকাউন্ট': 2.0,
            'একাউন্ট খুলতে': 2.0,
            'অ্যাকাউন্ট খোলার': 2.0,
            'সম্পর্কে জানতে চাই': 1.5,
            'সেবা সম্পর্কে': 1.5,
        }
        
        # ULTRA-PRECISION: Collection-specific phrase libraries
        self.collection_phrases = {
            'faq_yaqeen': {
                # Islamic Banking Specific Phrases
                'ইয়াকিন': 4.0,
                'ইয়াকিন সেভিংস': 4.0,
                'ইয়াকিন কারেন্ট': 4.0,
                'ইয়াকিন জুনিয়র': 4.0,
                'ইয়াকিন বানাত': 4.0,
                'ইয়াকিন অঘনিয়া': 4.0,
                'ইয়াকিন অশরিয়া': 4.0,
                'ইয়াকিন পেরোল': 4.0,
                'ইসলামিক ব্যাংকিং': 3.5,
                'শরিয়া সুপারভাইজরি': 3.5,
                'মুদারাবা': 3.0,
                'উজরা কার্ড': 3.5,
                'হালাল': 3.0,
                'মুরাবাহা': 3.0,
                'বাই সালাম': 3.0,
                'ইসলামিক একাউন্ট': 3.0,
                'শরীয়াহ': 3.0,
                'অঘনিয়া লাখপতি': 3.5,
                'অশরিয়া কোটিপতি': 3.5
            },
            'faq_retail': {
                # Conventional Banking Specific Phrases  
                'এমটিবি রেগুলার': 4.0,
                'এমটিবি ইন্সপায়ার': 4.0,
                'এমটিবি সিনিয়র': 4.0,
                'এমটিবি জুনিয়র': 4.0,
                'এমটিবি এক্সট্রিম': 4.0,
                'এমটিবি লাখপতি': 4.0,
                'এমটিবি কোটিপতি': 4.0,
                'এমটিবি মিলিয়নিয়ার': 4.0,
                'এমটিবি ব্রিক': 4.0,
                'রেগুলার সেভিংস': 3.5,
                'ইন্সপায়ার একাউন্ট': 3.5,
                'সিনিয়র একাউন্ট': 3.5,
                'ইন্টারেস্ট রেট': 3.0,
                'সুদের হার': 3.0,
                'এফডিআর': 3.0,
                'ডিপিএস': 3.0
            },
            'faq_payroll': {
                'পেরোল একাউন্ট': 4.0,
                'পেরোল ব্যাংকিং': 4.0,
                'স্যালারি একাউন্ট': 3.5,
                'বেতন একাউন্ট': 3.5,
                'পেরোল প্রিমিয়াম': 3.5,
                'পেরোল সেভার্স': 3.5,
                'পেরোল ইসেভার্স': 3.5
            },
            'faq_women': {
                'অঙ্গনা': 4.0,
                'অঙ্গনা একাউন্ট': 4.0,
                'নারী গ্রাহক': 3.5,
                'মহিলা একাউন্ট': 3.5,
                'বানাত': 3.5
            },
            'faq_sme': {
                'এসএমই': 4.0,
                'এসএমই লোন': 4.0,
                'ব্যবসায়িক ঋণ': 3.5,
                'উদ্যোক্তা': 3.0
            },
            'faq_card': {
                'ক্রেডিট কার্ড': 4.0,
                'ডেবিট কার্ড': 3.5,
                'ভিসা কার্ড': 3.5,
                'মাস্টার কার্ড': 3.5
            }
        }
        
        # ULTRA-PRECISION: Negative keywords (penalize wrong collections)
        self.negative_keywords = {
            'faq_yaqeen': {
                # Penalize if conventional banking terms appear in Islamic context
                'ইন্টারেস্ট': -2.0,
                'সুদের হার': -2.0,
                'এমটিবি রেগুলার': -1.5,
                'এমটিবি ইন্সপায়ার': -1.5
            },
            'faq_retail': {
                # Penalize if Islamic banking terms appear in conventional context
                'ইয়াকিন': -1.5,
                'ইসলামিক': -2.0,
                'শরিয়া': -2.0,
                'মুদারাবা': -2.0,
                'হালাল': -1.5
            }
        }
        
        # ULTRA-PRECISION: Context window phrases (word combinations)
        self.context_phrases = {
            'islamic_intent': [
                'ইয়াকিন.*সেভিংস',
                'ইসলামিক.*একাউন্ট', 
                'শরিয়া.*সম্মত',
                'মুদারাবা.*নীতি',
                'হালাল.*বিনিয়োগ'
            ],
            'conventional_intent': [
                'এমটিবি.*রেগুলার',
                'এমটিবি.*ইন্সপায়ার',
                'সুদের.*হার',
                'ইন্টারেস্ট.*রেট'
            ]
        }
        
        # ULTRA-PRECISION: Collection-specific keyword expansion
        self.keyword_expansions = {
            'faq_yaqeen': {
                'লাখপতি': ['অঘনিয়া লাখপতি', 'ইয়াকিন অঘনিয়া'],
                'কোটিপতি': ['অশরিয়া কোটিপতি', 'ইয়াকিন অশরিয়া'],
                'পেরোল': ['ইয়াকিন পেরোল', 'ইসলামিক পেরোল'],
                'একাউন্ট': ['ইয়াকিন একাউন্ট', 'ইসলামিক একাউন্ট']
            },
            'faq_retail': {
                'লাখপতি': ['এমটিবি লাখপতি'],
                'কোটিপতি': ['এমটিবি কোটিপতি'],
                'মিলিয়নিয়ার': ['এমটিবি মিলিয়নিয়ার'],
                'একাউন্ট': ['রেগুলার একাউন্ট', 'ইন্সপায়ার একাউন্ট']
            }
        }
        
        # ULTRA-PRECISION: Cross-collection penalty matrix
        self.cross_collection_penalties = {
            ('faq_yaqeen', 'faq_retail'): 0.8,  # High penalty for Islamic vs Conventional confusion
            ('faq_retail', 'faq_yaqeen'): 0.8,  # High penalty for Conventional vs Islamic confusion
            ('faq_women', 'faq_yaqeen'): 0.95,  # Low penalty (women's Islamic banking is valid)
            ('faq_payroll', 'faq_yaqeen'): 0.9, # Medium penalty (Islamic payroll exists)
        }
        
        # ULTRA-PRECISION: Collection-specific n-gram weights
        self.collection_ngram_weights = {
            'faq_yaqeen': {
                2: 0.4,  # Bigrams important for "ইয়াকিন সেভিংস"
                3: 0.6   # Trigrams very important for "ইয়াকিন অঘনিয়া লাখপতি"
            },
            'faq_retail': {
                2: 0.5,  # Bigrams important for "এমটিবি রেগুলার" 
                3: 0.5   # Equal importance for trigrams
            },
            'default': {
                2: 0.3,
                3: 0.7
            }
        }
            
        logger.info(f"Initialized HybridMatcher with weights: {self.weights}")
        logger.info(f"Loaded {len(self.banking_phrases)} banking phrases for matching")
    
    def calculate_collection_specific_similarity(self, 
                                               query: str, 
                                               faq_question: str,
                                               collection: str = None,
                                               cleaned_query: str = None, 
                                               cleaned_faq: str = None,
                                               embedding_similarity: float = None) -> dict:
        """
        ULTRA-PRECISION: Calculate collection-aware hybrid similarity
        """
        # Start with base similarity calculation
        result = self.calculate_similarity(
            query, faq_question, cleaned_query, cleaned_faq, embedding_similarity
        )
        
        if not collection:
            return result
            
        # ULTRA-PRECISION ENHANCEMENTS
        collection_boost = 0.0
        context_boost = 0.0
        negative_penalty = 0.0
        expansion_boost = 0.0
        collection_ngram_score = 0.0
        
        query_lower = query.lower()
        question_lower = faq_question.lower()
        
        # 1. Collection-specific phrase matching
        if collection in self.collection_phrases:
            for phrase, weight in self.collection_phrases[collection].items():
                if phrase in query_lower and phrase in question_lower:
                    collection_boost += weight * 0.1  # Strong boost for exact collection match
                elif phrase in query_lower or phrase in question_lower:
                    collection_boost += weight * 0.05  # Moderate boost for partial match
        
        # 2. Negative keyword penalties
        if collection in self.negative_keywords:
            for negative_term, penalty in self.negative_keywords[collection].items():
                if negative_term in query_lower:
                    negative_penalty += abs(penalty) * 0.1  # Convert to positive penalty
        
        # 3. Context window analysis
        context_boost += self._analyze_context_window(query_lower, question_lower, collection)
        
        # 4. Sequential pattern importance
        sequential_boost = self._calculate_sequential_importance(query_lower, question_lower, collection)
        
        # 5. ULTRA-PRECISION: Collection-aware n-gram similarity (replace base n-gram)
        collection_ngram_score = self._calculate_collection_aware_ngram_similarity(
            query_lower, question_lower, collection
        )
        if collection_ngram_score > result['ngram_match']:
            ngram_improvement = collection_ngram_score - result['ngram_match']
            collection_boost += ngram_improvement * 0.4  # Weight for n-gram improvement
        
        # 6. ULTRA-PRECISION: Keyword expansion matching
        expansion_boost = self._calculate_keyword_expansion_match(query, faq_question, collection)
        collection_boost += expansion_boost
        
        # Apply ultra-precision adjustments
        ultra_precision_adjustment = collection_boost + context_boost + sequential_boost - negative_penalty
        
        # Update the result with ultra-precision enhancements
        result['collection_boost'] = collection_boost
        result['context_boost'] = context_boost  
        result['negative_penalty'] = negative_penalty
        result['sequential_boost'] = sequential_boost
        result['expansion_boost'] = expansion_boost
        result['collection_ngram_score'] = collection_ngram_score
        result['ultra_precision_adjustment'] = ultra_precision_adjustment
        
        # Apply adjustment with diminishing returns
        headroom = 1.0 - result['final_score']
        max_adjustment = min(abs(ultra_precision_adjustment), headroom * 0.3)
        
        if ultra_precision_adjustment > 0:
            result['final_score'] += max_adjustment
        else:
            result['final_score'] = max(0.0, result['final_score'] - max_adjustment)
        
        return result
    
    def _analyze_context_window(self, query: str, question: str, collection: str) -> float:
        """Analyze context window for collection-specific intent"""
        import re
        
        boost = 0.0
        
        # Check for context patterns
        for intent, patterns in self.context_phrases.items():
            for pattern in patterns:
                if re.search(pattern, query) and re.search(pattern, question):
                    if intent == 'islamic_intent' and collection == 'faq_yaqeen':
                        boost += 0.15
                    elif intent == 'conventional_intent' and collection == 'faq_retail':
                        boost += 0.15
                    elif intent == 'islamic_intent' and collection != 'faq_yaqeen':
                        boost -= 0.1  # Penalty for wrong collection
                    elif intent == 'conventional_intent' and collection == 'faq_yaqeen':
                        boost -= 0.1  # Penalty for wrong collection
        
        return boost
    
    def _calculate_sequential_importance(self, query: str, question: str, collection: str) -> float:
        """Calculate importance of word sequence/order for precision"""
        words_query = query.split()
        words_question = question.split()
        
        if len(words_query) < 2 or len(words_question) < 2:
            return 0.0
        
        # Find longest common subsequence (preserving order)
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(words_query, words_question)
        
        # Calculate sequence importance score
        max_len = max(len(words_query), len(words_question))
        if max_len == 0:
            return 0.0
            
        sequence_score = lcs_len / max_len
        
        # Boost if important collection-specific terms appear in sequence
        important_sequences = {
            'faq_yaqeen': ['ইয়াকিন', 'ইসলামিক'],
            'faq_retail': ['রেগুলার'],
        }
        
        sequence_boost = 0.0
        if collection in important_sequences:
            important_terms = important_sequences[collection]
            for i in range(len(words_query) - 1):
                if words_query[i] in important_terms and words_query[i+1] in important_terms:
                    # Check if same sequence exists in question
                    for j in range(len(words_question) - 1):
                        if words_question[j] == words_query[i] and words_question[j+1] == words_query[i+1]:
                            sequence_boost += 0.1
                            break
        
        return sequence_score * 0.1 + sequence_boost
    
    def calculate_similarity(self, 
                           query: str, 
                           faq_question: str, 
                           cleaned_query: str = None, 
                           cleaned_faq: str = None,
                           embedding_similarity: float = None) -> dict:
        """
        Calculate hybrid similarity between query and FAQ question
        
        Returns a dict with the similarity score and detailed component scores
        """
        # Initialize result
        result = {
            'exact_match': 0.0,
            'cleaned_match': 0.0,
            'ngram_match': 0.0,
            'keyword_match': 0.0,
            'phrase_match': 0.0,
            'embedding': embedding_similarity or 0.0,
            'final_score': 0.0
        }
        
        # Perform exact string match check
        if query.lower() == faq_question.lower():
            result['exact_match'] = 1.0
        
        # Perform cleaned text match if cleaned versions provided
        if cleaned_query and cleaned_faq and cleaned_query.lower() == cleaned_faq.lower():
            result['cleaned_match'] = 1.0
            
        # Calculate n-gram similarity if not an exact match
        if result['exact_match'] < 1.0 and result['cleaned_match'] < 1.0:
            result['ngram_match'] = self._calculate_ngram_similarity(
                query.lower(), faq_question.lower()
            )
            
            # Calculate keyword similarity
            result['keyword_match'] = self._calculate_keyword_similarity(
                query.lower(), faq_question.lower()
            )
            
            # Calculate phrase-based matching for banking domain
            result['phrase_match'] = self._calculate_phrase_similarity(
                query.lower(), faq_question.lower()
            )
        
        # Calculate weighted score
        weighted_sum = sum([
            result['exact_match'] * self.weights['exact_match'],
            result['cleaned_match'] * self.weights['cleaned_match'],
            result['ngram_match'] * self.weights['ngram_match'],
            result['keyword_match'] * self.weights['keyword_match'],
            result['phrase_match'] * self.weights['phrase_match'],
            result['embedding'] * self.weights['embedding']
        ])
        
        weight_sum = sum([
            self.weights['exact_match'] if result['exact_match'] > 0 else 0,
            self.weights['cleaned_match'] if result['cleaned_match'] > 0 else 0,
            self.weights['ngram_match'],
            self.weights['keyword_match'],
            self.weights['phrase_match'] if result['phrase_match'] > 0 else 0,
            self.weights['embedding'] if embedding_similarity is not None else 0
        ])
        
        # Normalize
        if weight_sum > 0:
            result['final_score'] = weighted_sum / weight_sum
        else:
            result['final_score'] = 0.0
        
        # Special rules for guaranteeing 100% score for exact matches
        if result['exact_match'] == 1.0 or result['cleaned_match'] == 1.0:
            result['final_score'] = 1.0
            
        # Apply boosting for non-exact matches that have strong semantic or keyword similarity
        elif result['final_score'] > 0.3:
            # Get boost factor from config (default 0.15 if not specified)
            boost_factor = self.weights.get('boost_factor', 0.15)
            
            # Calculate component strength indicators
            strong_keywords = result['keyword_match'] >= 0.6
            good_embedding = result['embedding'] >= 0.6
            decent_ngrams = result['ngram_match'] >= 0.4
            
            # Define boost conditions and amounts
            boost_amount = 0
            
            # Strong keyword match deserves a boost
            if strong_keywords:
                boost_amount += boost_factor
                
            # Good embedding similarity deserves a boost
            if good_embedding:
                boost_amount += boost_factor
                
            # Decent n-gram similarity deserves a smaller boost
            if decent_ngrams:
                boost_amount += boost_factor * 0.5
                
            # Combination of factors deserves an extra boost
            if (strong_keywords and good_embedding) or \
               (strong_keywords and decent_ngrams) or \
               (good_embedding and decent_ngrams):
                boost_amount += boost_factor * 0.5
                
            # Apply the boost with diminishing returns as we approach 1.0
            headroom = 1.0 - result['final_score']
            adjusted_boost = min(boost_amount, headroom * 0.8)  # Cap at 80% of remaining headroom
            
            result['final_score'] += adjusted_boost
            
            # Add info about the boost applied
            result['boost_applied'] = adjusted_boost
            
        # Debug logging
        logger.debug(f"Hybrid matching: {result}")
            
        return result
    
    def _calculate_ngram_similarity(self, text1: str, text2: str, n_values: list = None) -> float:
        """Calculate n-gram similarity between two texts using multiple n-gram sizes"""
        if not text1 or not text2:
            return 0.0
        
        # Default to using bigrams and trigrams    
        if n_values is None:
            n_values = [2, 3]
            
        # Clean texts for more effective n-gram matching
        # Remove extra spaces and normalize Bengali characters
        t1 = re.sub(r'\s+', ' ', text1.lower().strip())
        t2 = re.sub(r'\s+', ' ', text2.lower().strip())
            
        # Generate n-grams for both texts for multiple values of n
        def get_ngrams(text, n):
            # For character-level n-grams
            char_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
            
            # For word-level n-grams (especially useful for Bengali)
            words = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', text)
            word_grams = []
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    word_grams.append(' '.join(words[i:i+n]))
                    
            return char_grams + word_grams
            
        # Calculate similarity for each n-gram size
        similarities = []
        importance_weights = {2: 0.3, 3: 0.7}  # Prefer trigrams as they capture more context
        
        for n in n_values:
            ngrams1 = set(get_ngrams(t1, n))
            ngrams2 = set(get_ngrams(t2, n))
            
            if not ngrams1 or not ngrams2:
                similarities.append(0.0)
                continue
                
            # Calculate weighted Jaccard similarity
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            # Calculate similarity
            sim = intersection / union if union > 0 else 0.0
            
            # Add sequence matching bonus
            # Check if there are any consecutive matching n-grams
            # This helps identify phrases that appear in both texts
            consecutive_matches = 0
            for i in range(len(t1) - n*2 + 1):
                chunk = t1[i:i+n*2]
                if chunk in t2:
                    consecutive_matches += 1
            
            seq_bonus = min(0.2, consecutive_matches * 0.05)  # Cap the bonus
            sim = min(1.0, sim + seq_bonus)  # Apply bonus but cap at 1.0
            
            # Store with weight
            weight = importance_weights.get(n, 0.5)
            similarities.append((sim, weight))
            
        # Combine similarities with weights
        if similarities:
            weighted_sum = sum(sim * weight for sim, weight in similarities)
            weight_sum = sum(weight for _, weight in similarities)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
        return 0.0
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword similarity between two texts with banking domain emphasis"""
        # Better tokenization for Bengali text (handle punctuation and whitespace better)
        words1 = set(re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', text1.lower()))
        words2 = set(re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Important banking domain terms with weights
        banking_terms = {
            'ব্যাংক': 3.0, 'ব্যাংকিং': 3.0, 'এটিএম': 2.5, 'কার্ড': 2.5,
            'লোন': 2.5, 'ঋণ': 2.5, 'একাউন্ট': 2.5, 'অ্যাকাউন্ট': 2.5, 
            'সেভিংস': 2.5, 'ইসলামিক': 3.0, 'ইয়াকিন': 3.0, 'শরিয়াহ': 3.0,
            'সার্ভিস': 2.0, 'সেবা': 2.0, 'চার্জ': 2.0, 'ফি': 2.0,
            'মোবাইল': 2.0, 'অনলাইন': 2.0, 'ইন্টারনেট': 2.0,
            'ডেবিট': 2.5, 'ক্রেডিট': 2.5, 'মুদারাবা': 3.0, 'শরীয়াহ': 3.0
        }
        
        # Weighted intersection calculation
        weighted_intersection = 0
        base_weight = 1.0
        
        for word in words1.intersection(words2):
            weight = banking_terms.get(word, base_weight)
            weighted_intersection += weight
        
        # Total possible weight
        total_weight = sum(banking_terms.get(w, base_weight) for w in words1.union(words2))
        
        # Add bonus for term coverage
        term_coverage = len(words1.intersection(words2)) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
        
        # Combine weighted and coverage scores
        similarity = (weighted_intersection / total_weight) if total_weight > 0 else 0
        combined_score = 0.7 * similarity + 0.3 * term_coverage
        
        return combined_score
        
    def _calculate_phrase_similarity(self, text1: str, text2: str) -> float:
        """Calculate phrase-based similarity for banking domain"""
        if not text1 or not text2:
            return 0.0
        
        # Clean the texts
        t1 = re.sub(r'\s+', ' ', text1.lower().strip())
        t2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Check for each banking phrase in both texts
        matches = []
        max_score = 0.0
        
        for phrase, weight in self.banking_phrases.items():
            # Check if phrase is in either text
            in_t1 = phrase in t1
            in_t2 = phrase in t2
            
            if in_t1 and in_t2:
                # Both texts contain this phrase - strong signal
                matches.append((phrase, weight, 1.0))
                max_score = max(max_score, 1.0)  # Direct match gets full score
            elif in_t1 or in_t2:
                # Only one text contains the phrase
                # Calculate distance to nearest similar phrase in the other text
                if in_t1:
                    source, target = t1, t2
                else:
                    source, target = t2, t1
                
                # Check if any part of the phrase appears in the target
                phrase_parts = phrase.split()
                if len(phrase_parts) > 1:
                    parts_found = sum(1 for part in phrase_parts if part in target)
                    if parts_found > 0:
                        partial_match = parts_found / len(phrase_parts)
                        score = weight * partial_match * 0.7  # Scale by 0.7 for partial match
                        matches.append((phrase, weight, score))
                        max_score = max(max_score, score)
        
        if not matches:
            return 0.0
            
        # Calculate weighted score from all matches
        total_weight = sum(w for _, w, _ in matches)
        weighted_score = sum(w * s for _, w, s in matches)
        
        # Normalize by total weight
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0
            
        # Factor in the max score to ensure a high score for important exact matches
        final_score = 0.7 * normalized_score + 0.3 * max_score
        
        return final_score
        
    def _calculate_collection_aware_ngram_similarity(self, text1: str, text2: str, collection: str = None, n_values: list = None) -> float:
        """ULTRA-PRECISION: Calculate collection-aware n-gram similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Get collection-specific n-gram weights
        if collection and collection in self.collection_ngram_weights:
            weights = self.collection_ngram_weights[collection]
        else:
            weights = self.collection_ngram_weights['default']
        
        # Default to using bigrams and trigrams    
        if n_values is None:
            n_values = [2, 3]
            
        # Clean texts for more effective n-gram matching
        t1 = re.sub(r'\s+', ' ', text1.lower().strip())
        t2 = re.sub(r'\s+', ' ', text2.lower().strip())
            
        # Generate n-grams for both texts
        def get_ngrams(text, n):
            # Character-level n-grams
            char_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
            
            # Word-level n-grams (especially useful for Bengali)
            words = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+', text)
            word_grams = []
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    word_grams.append(' '.join(words[i:i+n]))
                    
            return char_grams + word_grams
            
        # Calculate similarity for each n-gram size with collection-specific weights
        similarities = []
        
        for n in n_values:
            ngrams1 = set(get_ngrams(t1, n))
            ngrams2 = set(get_ngrams(t2, n))
            
            if not ngrams1 or not ngrams2:
                similarities.append((0.0, weights.get(n, 0.5)))
                continue
                
            # Calculate weighted Jaccard similarity
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            # Calculate base similarity
            sim = intersection / union if union > 0 else 0.0
            
            # ULTRA-PRECISION: Collection-specific n-gram importance
            if collection and n == 3:  # Trigrams are crucial for collection-specific phrases
                # Check for collection-specific trigram patterns
                collection_boost = 0.0
                if collection in self.collection_phrases:
                    for phrase in self.collection_phrases[collection]:
                        if phrase in t1 and phrase in t2:
                            collection_boost += 0.15  # Strong boost for collection-specific phrases
                            break
                sim = min(1.0, sim + collection_boost)
            
            # Add sequence matching bonus
            consecutive_matches = 0
            for i in range(len(t1) - n*2 + 1):
                chunk = t1[i:i+n*2]
                if chunk in t2:
                    consecutive_matches += 1
            
            seq_bonus = min(0.2, consecutive_matches * 0.05)
            sim = min(1.0, sim + seq_bonus)
            
            # Store with collection-specific weight
            weight = weights.get(n, 0.5)
            similarities.append((sim, weight))
            
        # Combine similarities with collection-aware weights
        if similarities:
            weighted_sum = sum(sim * weight for sim, weight in similarities)
            weight_sum = sum(weight for _, weight in similarities)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
            
        return 0.0
    
    def _calculate_keyword_expansion_match(self, query: str, question: str, collection: str) -> float:
        """ULTRA-PRECISION: Calculate keyword expansion matching for collection-specific terms"""
        if not collection or collection not in self.keyword_expansions:
            return 0.0
        
        query_lower = query.lower()
        question_lower = question.lower()
        expansion_boost = 0.0
        
        # Check keyword expansions for this collection
        for base_keyword, expansions in self.keyword_expansions[collection].items():
            if base_keyword in query_lower:
                # Check if any expansion appears in the question
                for expansion in expansions:
                    if expansion in question_lower:
                        expansion_boost += 0.2  # Strong boost for keyword expansion match
                        break
                        
        return min(0.5, expansion_boost)  # Cap the expansion boost


class IntentConfusionMatrix:
    """
    Track and analyze FAQ matching performance
    Provides confusion matrix and accuracy metrics
    """
    
    def __init__(self, faq_questions: List[str] = None):
        """Initialize with a list of FAQ questions"""
        self.faq_questions = faq_questions or []
        self.confusion_matrix = {}
        self.queries = []
        
    def add_result(self, query: str, expected: str, matched: str, score: float):
        """Add a matching result to the confusion matrix"""
        if query not in self.queries:
            self.queries.append(query)
            
        key = (query, expected, matched)
        self.confusion_matrix[key] = score
        
    def get_matrix(self):
        """Return the confusion matrix data"""
        return self.confusion_matrix
        
    def get_accuracy(self):
        """Calculate accuracy metrics"""
        total = len(self.confusion_matrix)
        if total == 0:
            return 0.0
            
        correct = sum(1 for k, v in self.confusion_matrix.items() 
                     if k[1] == k[2] and v >= 0.6)  # Threshold for "correct" match
                     
        return correct / total if total > 0 else 0.0


def hybrid_enhance_candidates(candidates: List[Dict], 
                             query: str, 
                             cleaned_query: str,
                             hybrid_matcher: HybridMatcher) -> List[Dict]:
    """
    Enhance a list of candidates with hybrid similarity scores using ultra-precision collection-aware matching
    
    This is a helper function to integrate with the existing FAQ service.
    """
    if not candidates:
        return []
        
    # Process each candidate with hybrid matching
    for candidate in candidates:
        faq_question = candidate.get("question", "")
        embedding_score = candidate.get("score", 0.0)
        collection = candidate.get("collection", "")
        
        # Calculate ultra-precision collection-aware hybrid score
        hybrid_result = hybrid_matcher.calculate_collection_specific_similarity(
            query=query,
            faq_question=faq_question,
            collection=collection,
            cleaned_query=cleaned_query,
            cleaned_faq=faq_question,  # Assume FAQ is already cleaned in DB
            embedding_similarity=embedding_score
        )
        
        # Update score with final hybrid score
        candidate["original_embedding_score"] = embedding_score
        candidate["score"] = hybrid_result["final_score"]
        candidate["match_details"] = hybrid_result
        
    # Resort based on new scores
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates
