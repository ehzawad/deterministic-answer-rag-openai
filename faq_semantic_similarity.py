#!/usr/bin/env python3
"""
FAQ Semantic Similarity Analysis and Confusion Matrix Generator

This script analyzes semantic similarity between different FAQ collections
and generates confusion matrices to visualize cross-domain similarities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

class FAQSemanticAnalyzer:
    def __init__(self, faq_data_dir='faq_data'):
        self.faq_data_dir = faq_data_dir
        self.collections = {}
        self.collection_names = []
        self.similarity_matrix = None
        
    def load_faq_collections(self):
        """Load all FAQ collections from the data directory."""
        for filename in sorted(os.listdir(self.faq_data_dir)):
            if filename.endswith('.txt'):
                collection_name = filename.replace('.txt', '').replace('_', ' ').title()
                self.collection_names.append(collection_name)
                
                file_path = os.path.join(self.faq_data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract questions and answers
                questions = []
                answers = []
                
                # Split by lines and extract Q&A pairs
                lines = content.split('\n')
                current_question = ""
                current_answer = ""
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Question:'):
                        if current_question and current_answer:
                            questions.append(current_question)
                            answers.append(current_answer)
                        current_question = line.replace('Question:', '').strip()
                        current_answer = ""
                    elif line.startswith('Answer:'):
                        current_answer = line.replace('Answer:', '').strip()
                    elif current_answer and line:
                        current_answer += " " + line
                
                # Add the last Q&A pair
                if current_question and current_answer:
                    questions.append(current_question)
                    answers.append(current_answer)
                
                # Combine questions and answers for comprehensive text analysis
                full_text = ' '.join(questions + answers)
                
                self.collections[collection_name] = {
                    'questions': questions,
                    'answers': answers,
                    'full_text': full_text,
                    'filename': filename
                }
        
        print(f"Loaded {len(self.collections)} FAQ collections:")
        for name in self.collection_names:
            q_count = len(self.collections[name]['questions'])
            print(f"  - {name}: {q_count} Q&A pairs")
    
    def calculate_similarity_matrix(self):
        """Calculate semantic similarity between collections using TF-IDF and cosine similarity."""
        print("\nCalculating semantic similarities...")
        
        # Prepare documents (full text from each collection)
        documents = [self.collections[name]['full_text'] for name in self.collection_names]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Keep Bengali words
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
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
        print(f"Confusion matrix saved to: {save_path}")
        
        return save_path
    
    def generate_detailed_report(self, save_path='faq_confusion_report.json'):
        """Generate detailed similarity analysis report."""
        print("\nGenerating detailed similarity report...")
        
        report = {
            'metadata': {
                'total_collections': len(self.collection_names),
                'collection_names': self.collection_names,
                'analysis_method': 'TF-IDF + Cosine Similarity'
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
            collection = self.collections[name]
            report['collection_details'][name] = {
                'filename': collection['filename'],
                'question_count': len(collection['questions']),
                'avg_question_length': np.mean([len(q.split()) for q in collection['questions']]) if collection['questions'] else 0,
                'avg_answer_length': np.mean([len(a.split()) for a in collection['answers']]) if collection['answers'] else 0,
                'total_words': len(collection['full_text'].split())
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
        
        # Cross-domain insights
        banking_related = [name for name in self.collection_names if any(word in name.lower() for word in ['banking', 'agent', 'sme', 'nrb', 'privilege'])]
        product_related = [name for name in self.collection_names if any(word in name.lower() for word in ['card', 'retails', 'payroll'])]
        islamic_related = [name for name in self.collection_names if 'yaqeen' in name.lower()]
        
        report['similarity_analysis']['cross_domain_insights'] = {
            'banking_services': banking_related,
            'product_services': product_related, 
            'islamic_services': islamic_related,
            'domain_separation_quality': self._analyze_domain_separation()
        }
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed report saved to: {save_path}")
        return report
    
    def _analyze_domain_separation(self):
        """Analyze how well different domains are separated."""
        insights = {}
        
        # Check Islamic vs Conventional banking separation
        yaqeen_idx = None
        conventional_banking_indices = []
        
        for i, name in enumerate(self.collection_names):
            if 'yaqeen' in name.lower():
                yaqeen_idx = i
            elif any(word in name.lower() for word in ['banking', 'agent', 'sme', 'nrb']):
                conventional_banking_indices.append(i)
        
        if yaqeen_idx is not None and conventional_banking_indices:
            yaqeen_conventional_similarities = [self.similarity_matrix[yaqeen_idx][idx] for idx in conventional_banking_indices]
            insights['islamic_conventional_separation'] = {
                'avg_similarity': float(np.mean(yaqeen_conventional_similarities)),
                'separation_quality': 'Good' if np.mean(yaqeen_conventional_similarities) < 0.5 else 'Moderate' if np.mean(yaqeen_conventional_similarities) < 0.7 else 'Poor'
            }
        
        return insights
    
    def print_summary(self):
        """Print a summary of the analysis."""
        print(f"\n{'='*60}")
        print("FAQ SEMANTIC SIMILARITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total Collections Analyzed: {len(self.collection_names)}")
        print(f"Collections: {', '.join(self.collection_names)}")
        
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
        print(f"  {max_pair[0]} ↔ {max_pair[1]} (Similarity: {max_similarity:.3f})")
        
        print(f"\nLeast Similar Collections:")
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
    analyzer.load_faq_collections()
    
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