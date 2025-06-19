#!/usr/bin/env python3
"""
🇧🇩 Advanced Bengali Semantic Engine
Deep linguistic understanding for nuanced Bengali banking queries
Handles morphology, syntax, cultural context, and semantic equivalences
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class BengaliMorpheme:
    """Bengali morpheme with grammatical information"""
    root: str
    prefix: str = ""
    suffix: str = ""
    pos_tag: str = ""  # Part of speech
    semantic_class: str = ""

@dataclass
class SemanticMapping:
    """Advanced semantic mapping for Bengali terms"""
    source_terms: List[str]
    target_terms: List[str]
    confidence: float
    context: str
    morphological_variants: List[str]

class BengaliSemanticEngine:
    """🧠 Advanced Bengali semantic understanding engine"""
    
    def __init__(self):
        self.morphological_rules = self._load_morphological_rules()
        self.semantic_mappings = self._load_semantic_mappings()
        self.banking_ontology = self._load_banking_ontology()
        self.cultural_context = self._load_cultural_context()
        self.phonetic_variants = self._load_phonetic_variants()
        
    def _load_morphological_rules(self) -> Dict:
        """🔤 Load Bengali morphological analysis rules"""
        return {
            'possessive_markers': ['এর', 'র', 'কে', 'তে', 'থেকে'],
            'verb_inflections': ['তে পারি', 'তে হবে', 'তে চাই', 'করতে', 'খুলতে', 'পেতে'],
            'question_words': ['কি', 'কী', 'কত', 'কেমন', 'কোন', 'কার', 'কেন', 'কিভাবে'],
            'respect_markers': ['গুলো', 'গুলি', 'সমূহ', 'দের']
        }
    
    def _load_semantic_mappings(self) -> Dict:
        """🎯 Advanced semantic mappings for Bengali banking"""
        return {
            # Account variations with morphological awareness
            'account_terms': {
                'patterns': [
                    {'input': r'(একাউন্ট|অ্যাকাউন্ট|হিসাব|খাতা)(এর|র|ে)?', 'output': 'একাউন্ট'},
                    {'input': r'(বেতনের|পেরোল|স্যালারি)(.*)(একাউন্ট|অ্যাকাউন্ট)', 'output': 'পেরোল একাউন্ট'},
                    {'input': r'(এমটিবি|MTB).*(রেগুলার|regular)', 'output': 'এমটিবি এনআরবি'},
                    {'input': r'(এসএমই|SME).*(একাউন্ট|অ্যাকাউন্ট)', 'output': 'এসএমই একাউন্ট'},
                    {'input': r'(মহিলা|নারী|অঙ্গনা).*(একাউন্ট|অ্যাকাউন্ট)', 'output': 'অঙ্গনা একাউন্ট'}
                ]
            },
            
            # Interest/Profit with Islamic context awareness
            'interest_terms': {
                'patterns': [
                    {'input': r'(সুদ|সুদের হার)', 'output': 'ইন্টারেস্ট রেট', 'context': 'conventional'},
                    {'input': r'(ইন্টারেস্ট|ইন্টারসেট).*(রেট|হার)', 'output': 'ইন্টারেস্ট রেট'},
                    {'input': r'(প্রফিট|লাভ).*(রেট|হার)', 'output': 'প্রফিট রেট', 'context': 'islamic'},
                    {'input': r'(ইয়াকিন|yaqeen).*(রেট|হার)', 'output': 'প্রফিট রেট', 'context': 'islamic'}
                ]
            },
            
            # Charges and fees with contextual mapping
            'charge_terms': {
                'patterns': [
                    {'input': r'(চার্জ|ফি|খরচ|ব্যয়)', 'output': 'চার্জ'},
                    {'input': r'(বাৎসরিক|বার্ষিক|yearly).*(চার্জ|ফি)', 'output': 'বার্ষিক চার্জ'},
                    {'input': r'(মাসিক|monthly).*(চার্জ|ফি)', 'output': 'মাসিক চার্জ'},
                    {'input': r'(কার্ড|card).*(চার্জ|ফি)', 'output': 'কার্ড চার্জ'}
                ]
            },
            
            # Benefits with comprehensive coverage
            'benefit_terms': {
                'patterns': [
                    {'input': r'(সুবিধা|সুবিধাসমূহ|বেনিফিট)', 'output': 'সুবিধা'},
                    {'input': r'(কি|কী).*(সুবিধা)', 'output': 'কি সুবিধা'},
                    {'input': r'(সুবিধা).*(কি|কী)', 'output': 'সুবিধা কি'},
                    {'input': r'(ফায়দা|লাভ|advantage)', 'output': 'সুবিধা'}
                ]
            }
        }
    
    def _load_banking_ontology(self) -> Dict:
        """🏦 Load comprehensive Bengali banking domain ontology"""
        return {
            'account_types': {
                'সঞ্চয়': 'savings',
                'চলতি': 'current', 
                'জমা': 'deposit',
                'মেয়াদী': 'fixed_deposit',
                'পেরোল': 'payroll',
                'এসএমই': 'sme',
                'কর্পোরেট': 'corporate',
                'অঙ্গনা': 'women_banking',
                'ইয়াকিন': 'islamic_banking',
                'প্রিভিলেজ': 'privilege'
            },
            
            'banking_actions': {
                'খোলা': 'open',
                'খুলতে': 'to_open', 
                'বন্ধ': 'close',
                'জমা': 'deposit',
                'তোলা': 'withdraw',
                'স্থানান্তর': 'transfer',
                'পাঠানো': 'send',
                'আনা': 'bring',
                'রাখা': 'keep'
            },
            
            'financial_terms': {
                'টাকা': 'money',
                'পয়সা': 'money',
                'অর্থ': 'money',
                'ব্যালেন্স': 'balance',
                'জমার পরিমাণ': 'deposit_amount',
                'মিনিমাম': 'minimum',
                'সর্বোচ্চ': 'maximum',
                'লিমিট': 'limit',
                'সীমা': 'limit'
            },
            
            'time_expressions': {
                'দৈনিক': 'daily',
                'সাপ্তাহিক': 'weekly', 
                'মাসিক': 'monthly',
                'বাৎসরিক': 'yearly',
                'বার্ষিক': 'annual',
                'তাৎক্ষণিক': 'instant',
                'অবিলম্বে': 'immediate'
            }
        }
    
    def _load_cultural_context(self) -> Dict:
        """🌟 Bengali cultural and banking context"""
        return {
            'islamic_indicators': ['ইয়াকিন', 'ইসলামিক', 'শরিয়া', 'হালাল', 'প্রফিট'],
            'conventional_indicators': ['এমটিবি', 'রেগুলার', 'সাধারণ', 'সুদ'],
            'business_context': ['এসএমই', 'ব্যবসা', 'ব্যবসায়ী', 'কোম্পানি', 'ফার্ম'],
            'personal_context': ['ব্যক্তিগত', 'পারিবারিক', 'নিজের'],
            'formality_markers': ['আপনি', 'আপনার', 'আপনাদের'],
            'question_intentions': {
                'information_seeking': ['জানতে চাই', 'বলুন', 'বলেন'],
                'decision_making': ['কোনটা ভালো', 'কোনটি উত্তম'],
                'problem_solving': ['সমস্যা', 'অসুবিধা', 'কষ্ট']
            }
        }
    
    def _load_phonetic_variants(self) -> Dict:
        """🔊 Load phonetic variants and common misspellings"""
        return {
            'phonetic_mappings': {
                'একাউন্ট': ['অ্যাকাউন্ট', 'একাউন্ট', 'একাউন্ট'],
                'ইন্টারেস্ট': ['ইন্টারসেট', 'ইন্টারেষ্ট', 'ইন্টেরেস্ট'],
                'চার্জ': ['চার্জ', 'চার্জ', 'চার্জ'],
                'সার্ভিস': ['সেবা', 'সার্ভিস', 'সেবিস'],
                'ব্যাংক': ['ব্যাঙ্ক', 'ব্যাংক', 'বাংক']
            },
            
            'common_typos': {
                'একাউন্ট': ['একাঊন্ট', 'একাউন্ত', 'একাউনট'],
                'সুবিধা': ['সুভিধা', 'সুবিদা', 'সুবিধ'],
                'চার্জ': ['চার্য', 'চার্জ', 'চার্জ']
            }
        }
    
    def analyze_query(self, query: str) -> Dict:
        """🧠 Comprehensive Bengali query analysis"""
        analysis = {
            'original': query,
            'morphology': self._analyze_morphology(query),
            'semantics': self._analyze_semantics(query),
            'context': self._analyze_context(query),
            'enhanced_query': self._enhance_query(query),
            'confidence_boosts': self._calculate_boosts(query)
        }
        return analysis
    
    def _analyze_morphology(self, query: str) -> Dict:
        """🔤 Morphological analysis"""
        analysis = {
            'question_type': None,
            'has_possessive': False,
            'has_verb_inflection': False,
            'politeness_level': 'neutral'
        }
        
        # Question type detection
        for qword in self.morphological_rules['question_words']:
            if qword in query:
                if qword in ['কত', 'কেমন']:
                    analysis['question_type'] = 'quantity_quality'
                elif qword in ['কি', 'কী']:
                    analysis['question_type'] = 'yes_no_what'
                elif qword in ['কিভাবে', 'কেন']:
                    analysis['question_type'] = 'method_reason'
                break
        
        # Possessive markers
        analysis['has_possessive'] = any(marker in query for marker in self.morphological_rules['possessive_markers'])
        
        # Verb inflections
        analysis['has_verb_inflection'] = any(verb in query for verb in self.morphological_rules['verb_inflections'])
        
        # Politeness level
        if any(formal in query for formal in ['আপনি', 'আপনার', 'আপনাদের']):
            analysis['politeness_level'] = 'formal'
        elif any(informal in query for informal in ['তুমি', 'তোমার']):
            analysis['politeness_level'] = 'informal'
        
        return analysis
    
    def _analyze_semantics(self, query: str) -> Dict:
        """🎯 Semantic pattern matching"""
        semantic_matches = {
            'account_related': [],
            'interest_related': [],
            'charge_related': [],
            'benefit_related': [],
            'confidence_scores': {}
        }
        
        # Apply pattern matching for each category
        for category, patterns_data in self.semantic_mappings.items():
            category_key = category.replace('_terms', '_related')
            if category_key in semantic_matches:
                for pattern_info in patterns_data['patterns']:
                    if re.search(pattern_info['input'], query, re.IGNORECASE):
                        match_info = {
                            'pattern': pattern_info['input'],
                            'replacement': pattern_info['output'],
                            'context': pattern_info.get('context', 'general')
                        }
                        semantic_matches[category_key].append(match_info)
                
                # Calculate confidence for this category
                semantic_matches['confidence_scores'][category_key] = len(semantic_matches[category_key]) * 0.2
        
        return semantic_matches
    
    def _analyze_context(self, query: str) -> Dict:
        """🌟 Cultural and domain context analysis"""
        context = {
            'banking_type': 'conventional',
            'domain_context': [],
            'formality': 'standard',
            'intent': 'information_seeking'
        }
        
        # Banking type detection
        if any(indicator in query.lower() for indicator in self.cultural_context['islamic_indicators']):
            context['banking_type'] = 'islamic'
        elif any(indicator in query.lower() for indicator in self.cultural_context['conventional_indicators']):
            context['banking_type'] = 'conventional'
        
        # Domain context
        if any(term in query.lower() for term in self.cultural_context['business_context']):
            context['domain_context'].append('business')
        if any(term in query.lower() for term in self.cultural_context['personal_context']):
            context['domain_context'].append('personal')
        
        # Formality
        if any(marker in query for marker in self.cultural_context['formality_markers']):
            context['formality'] = 'high'
        
        # Intent detection
        for intent, markers in self.cultural_context['question_intentions'].items():
            if any(marker in query for marker in markers):
                context['intent'] = intent
                break
        
        return context
    
    def _enhance_query(self, query: str) -> str:
        """🚀 Generate semantically enhanced query"""
        enhanced = query
        semantics = self._analyze_semantics(query)
        
        # Apply high-confidence semantic replacements
        for category, matches in semantics.items():
            if category == 'confidence_scores':
                continue
            
            for match in matches:
                if semantics['confidence_scores'].get(category, 0) > 0.3:
                    # Apply pattern replacement
                    enhanced = re.sub(match['pattern'], match['replacement'], enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _calculate_boosts(self, query: str) -> Dict:
        """📊 Calculate confidence boosts"""
        boosts = {
            'morphological': 0.0,
            'semantic': 0.0,
            'contextual': 0.0,
            'total': 0.0
        }
        
        morphology = self._analyze_morphology(query)
        semantics = self._analyze_semantics(query)
        context = self._analyze_context(query)
        
        # Morphological boost
        if morphology['question_type']:
            boosts['morphological'] += 0.1
        if morphology['has_possessive']:
            boosts['morphological'] += 0.05
        if morphology['politeness_level'] == 'formal':
            boosts['morphological'] += 0.05
        
        # Semantic boost
        total_semantic_confidence = sum(semantics['confidence_scores'].values())
        boosts['semantic'] = min(0.3, total_semantic_confidence)
        
        # Contextual boost
        if context['banking_type'] != 'conventional':
            boosts['contextual'] += 0.1
        if context['domain_context']:
            boosts['contextual'] += 0.05 * len(context['domain_context'])
        if context['formality'] == 'high':
            boosts['contextual'] += 0.05
        
        boosts['total'] = sum([boosts['morphological'], boosts['semantic'], boosts['contextual']])
        
        return boosts
    
    def optimize_for_search(self, query: str) -> Tuple[str, Dict]:
        """🔍 Optimize query for search with metadata"""
        analysis = self.analyze_query(query)
        
        optimized_query = analysis['enhanced_query']
        metadata = {
            'confidence_boost': analysis['confidence_boosts']['total'],
            'semantic_confidence': analysis['confidence_boosts']['semantic'],
            'question_type': analysis['morphology']['question_type'],
            'banking_context': analysis['context']['banking_type'],
            'formality': analysis['context']['formality']
        }
        
        return optimized_query, metadata

def main():
    """🧪 Test the semantic engine"""
    engine = BengaliSemanticEngine()
    
    test_queries = [
        "পেরোল একাউন্টের সুবিধাসমূহ কী কী?",
        "এমটিবি রেগুলার অ্যাকাউন্টে কি চার্জ আছে?", 
        "বেতনের একাউন্টে সুবিধা কি?",
        "ইয়াকিন একাউন্টের ইন্টারেস্ট রেট কত?",
        "মহিলাদের জন্য বিশেষ একাউন্ট আছে কি?"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Query: {query}")
        optimized, metadata = engine.optimize_for_search(query)
        print(f"🚀 Optimized: {optimized}")
        print(f"📊 Confidence boost: {metadata['confidence_boost']:.3f}")
        print(f"❓ Question type: {metadata['question_type']}")
        print(f"🏦 Banking context: {metadata['banking_context']}")

if __name__ == "__main__":
    main() 