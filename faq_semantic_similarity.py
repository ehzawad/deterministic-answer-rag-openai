#!/usr/bin/env python3
"""
FAQ Semantic Similarity Analysis Tool

This script analyzes semantic relationships between FAQ collections,
integrated with the main faq_service for consistency.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple
import logging

# Import the faq_service to reuse existing functionality
from faq_service import faq_service, FAQ_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAQSemanticAnalyzer:
    def __init__(self):
        self.collection_names = []
        self.questions_by_collection = defaultdict(list)
        self.embeddings_by_collection = defaultdict(list)
        
    def load_faq_collections_from_service(self):
        """Load FAQ collections using the existing faq_service."""
        try:
            # Ensure FAQ service is initialized
            if not faq_service.initialized:
                logger.info("Initializing FAQ service...")
                if not faq_service.initialize():
                    raise RuntimeError("Failed to initialize FAQ service")
            
            # Get all collections from the service
            stats = faq_service.get_system_stats()
            collections = stats.get('collections', {})
            
            for collection_name, collection_info in collections.items():
                # Extract meaningful collection name
                display_name = collection_name.replace('faq_', '').replace('_', ' ').title()
                self.collection_names.append(display_name)
                
                # Get questions from this collection
                # Since we can't directly access the ChromaDB collections,
                # we'll use the file parsing approach but with consistency
                filename = collection_name.replace('faq_', '') + '.txt'
                filepath = os.path.join(FAQ_DIR, filename)
                
                if os.path.exists(filepath):
                    faq_pairs = faq_service._preprocess_faq_file(filepath)
                    questions = [pair['question'] for pair in faq_pairs]
                    self.questions_by_collection[display_name] = questions
                    
                    # Generate embeddings using the service
                    if questions and not faq_service.test_mode:
                        embeddings = faq_service._create_embeddings(questions)
                        self.embeddings_by_collection[display_name] = embeddings
                        
            logger.info(f"Loaded {len(self.collection_names)} collections")
            
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            raise

    def calculate_similarity_matrix(self):
        """Calculate semantic similarity between collections using embeddings."""
        logger.info("Calculating semantic similarities...")
        
        if faq_service.test_mode:
            logger.warning("Running in test mode - using dummy similarities")
            # Create a dummy similarity matrix for test mode
            n = len(self.collection_names)
            self.similarity_matrix = np.eye(n) * 0.8 + np.random.rand(n, n) * 0.2
            return self.similarity_matrix
        
        # Calculate average embeddings for each collection
        collection_embeddings = []
        for name in self.collection_names:
            embeddings = self.embeddings_by_collection.get(name, [])
            if embeddings:
                # Average all question embeddings to get collection embedding
                avg_embedding = np.mean(embeddings, axis=0)
                collection_embeddings.append(avg_embedding)
            else:
                # Fallback for empty collections
                collection_embeddings.append(np.zeros(1024))  # EMBEDDING_DIMENSIONS
        
        # Calculate cosine similarity matrix
        n = len(collection_embeddings)
        self.similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.similarity_matrix[i][j] = 1.0
                else:
                    # Cosine similarity
                    dot_product = np.dot(collection_embeddings[i], collection_embeddings[j])
                    norm_i = np.linalg.norm(collection_embeddings[i])
                    norm_j = np.linalg.norm(collection_embeddings[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        self.similarity_matrix[i][j] = dot_product / (norm_i * norm_j)
                    else:
                        self.similarity_matrix[i][j] = 0.0
        
        return self.similarity_matrix
    
    def generate_confusion_matrix_plot(self, save_path='faq_semantic_analysis.png'):
        """Generate and save confusion matrix visualization."""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            self.similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            square=True,
            linewidths=0.5,
            xticklabels=self.collection_names,
            yticklabels=self.collection_names,
            cbar_kws={"shrink": 0.8},
            vmin=0,
            vmax=1
        )
        
        plt.title('FAQ Collections Semantic Similarity Matrix\n(Higher values indicate more similar content)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('FAQ Collections', fontweight='bold')
        plt.ylabel('FAQ Collections', fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
        
        return save_path
    
    def generate_detailed_report(self, save_path='faq_confusion_report.json'):
        """Generate detailed similarity analysis report."""
        logger.info("Generating detailed similarity report...")
        
        report = {
            'metadata': {
                'total_collections': len(self.collection_names),
                'collection_names': self.collection_names,
                'analysis_method': 'OpenAI Embeddings + Cosine Similarity' if not faq_service.test_mode else 'Test Mode (Dummy Data)'
            },
            'collection_details': {},
            'similarity_analysis': {
                'most_similar_pairs': [],
                'least_similar_pairs': [],
                'average_similarities': {},
                'cross_domain_insights': {}
            }
        }
        
        # Collection details
        for name in self.collection_names:
            questions = self.questions_by_collection.get(name, [])
            report['collection_details'][name] = {
                'question_count': len(questions),
                'avg_question_length': np.mean([len(q.split()) for q in questions]) if questions else 0
            }
        
        # Similarity analysis
        similarities = []
        for i in range(len(self.collection_names)):
            for j in range(i+1, len(self.collection_names)):
                similarity = self.similarity_matrix[i][j]
                similarities.append({
                    'collection_1': self.collection_names[i],
                    'collection_2': self.collection_names[j],
                    'similarity_score': float(similarity)
                })
                
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Most and least similar pairs
        report['similarity_analysis']['most_similar_pairs'] = similarities[:5]
        report['similarity_analysis']['least_similar_pairs'] = similarities[-5:]
        
        # Average similarity for each collection
        for i, name in enumerate(self.collection_names):
            # Calculate average similarity with all other collections
            other_similarities = [self.similarity_matrix[i][j] for j in range(len(self.collection_names)) if i != j]
            report['similarity_analysis']['average_similarities'][name] = {
                'avg_similarity': float(np.mean(other_similarities)),
                'max_similarity': float(np.max(other_similarities)),
                'min_similarity': float(np.min(other_similarities))
            }
        
        # Save report
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed report saved to: {save_path}")
        return report
    
    def print_summary(self):
        """Print a summary of the analysis."""
        print(f"\n{'='*60}")
        print("FAQ SEMANTIC SIMILARITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total Collections Analyzed: {len(self.collection_names)}")
        print(f"Collections: {', '.join(self.collection_names)}")
        
        if hasattr(self, 'similarity_matrix'):
            # Find most and least similar pairs
            max_similarity = 0
            min_similarity = 1
            max_pair = None
            min_pair = None
            
            for i in range(len(self.collection_names)):
                for j in range(i+1, len(self.collection_names)):
                    similarity = self.similarity_matrix[i][j]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_pair = (self.collection_names[i], self.collection_names[j])
                    if similarity < min_similarity:
                        min_similarity = similarity
                        min_pair = (self.collection_names[i], self.collection_names[j])
            
            print(f"\nMost Similar Collections:")
            if max_pair:
                print(f"  {max_pair[0]} ↔ {max_pair[1]} (Similarity: {max_similarity:.3f})")
            
            print(f"\nLeast Similar Collections:")
            if min_pair:
                print(f"  {min_pair[0]} ↔ {min_pair[1]} (Similarity: {min_similarity:.3f})")
            
            # Overall statistics
            upper_triangle = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
            print(f"\nOverall Similarity Statistics:")
            print(f"  Average Similarity: {np.mean(upper_triangle):.3f}")
            print(f"  Standard Deviation: {np.std(upper_triangle):.3f}")
            print(f"  Range: {np.min(upper_triangle):.3f} - {np.max(upper_triangle):.3f}")

def main():
    """Main execution function."""
    print("Starting FAQ Semantic Similarity Analysis...")
    
    # Initialize analyzer
    analyzer = FAQSemanticAnalyzer()
    
    # Load FAQ collections
    analyzer.load_faq_collections_from_service()
    
    # Calculate similarity matrix
    analyzer.calculate_similarity_matrix()
    
    # Generate visualizations and reports
    plot_path = analyzer.generate_confusion_matrix_plot()
    report_path = analyzer.generate_detailed_report()
    
    # Print summary
    analyzer.print_summary()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Confusion Matrix: {plot_path}")
    print(f"Detailed Report: {report_path}")
    print("\nUse these files to understand semantic relationships between your FAQ collections.")

if __name__ == "__main__":
    main()