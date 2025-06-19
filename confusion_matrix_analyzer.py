#!/usr/bin/env python3
"""
Confusion Matrix Analyzer for Bengali FAQ System
Analyzes semantic understanding performance and creates detailed confusion matrices
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os
from faq_service import faq_service

class FAQConfusionMatrix:
    """Comprehensive confusion matrix analysis for FAQ system"""
    
    def __init__(self):
        self.results = []
        self.collection_mapping = {
            'faq_payroll': 'Payroll',
            'faq_yaqeen': 'Islamic Banking', 
            'faq_retail': 'Retail Banking',
            'faq_sme': 'SME Banking',
            'faq_card': 'Cards',
            'faq_women': 'Women Banking',
            'faq_privilege': 'Privilege',
            'faq_agent': 'Agent Banking',
            'faq_nrb': 'NRB Banking'
        }
        
    def add_result(self, query: str, expected_collection: str, predicted_collection: str, 
                   expected_question: str, predicted_question: str, confidence: float, 
                   found: bool, semantic_type: str = "general"):
        """Add a test result to the confusion matrix"""
        result = {
            'query': query,
            'expected_collection': expected_collection,
            'predicted_collection': predicted_collection,
            'expected_question': expected_question,
            'predicted_question': predicted_question,
            'confidence': confidence,
            'found': found,
            'semantic_type': semantic_type,
            'collection_match': expected_collection == predicted_collection,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
    
    def create_test_cases(self) -> List[Dict]:
        """Create comprehensive test cases covering all collections and semantic types"""
        test_cases = [
            # Payroll Banking - Various semantic variations
            {
                'query': 'পেরোল অ্যাকাউন্টের সুদের হার কত?',
                'expected_collection': 'faq_payroll',
                'expected_question': 'পেরোল অ্যাকাউন্ট এর ইন্টারসেট রেট কত?',
                'semantic_type': 'interest_rate_equivalence'
            },
            {
                'query': 'পেরোল একাউন্টের সুবিধাসমূহ কী কী?',
                'expected_collection': 'faq_payroll',
                'expected_question': 'পেরোল একাউন্ট এর কি কি সুবিধা আছে?',
                'semantic_type': 'benefits_query'
            },
            {
                'query': 'স্যালারি একাউন্ট খুলতে কি ডকুমেন্ট লাগে?',
                'expected_collection': 'faq_payroll',
                'expected_question': 'আমি এমটিবির পেরোল একাউন্ট ওপেন করতে চাই, কি কি ডকুমেন্ট লাগবে?',
                'semantic_type': 'payroll_synonym'
            },
            {
                'query': 'পেরোল কার্ডের ফি কেমন?',
                'expected_collection': 'faq_payroll',
                'expected_question': 'পেরোল একাউন্ট এ ডেবিট কার্ড এর চার্জ আছে কি?',
                'semantic_type': 'fee_charge_equivalence'
            },
            
            # Islamic Banking (Yaqeen) - Complex semantic cases
            {
                'query': 'ইয়াকিন একাউন্টের ইন্টারেস্ট রেট কত?',
                'expected_collection': 'faq_yaqeen',
                'expected_question': 'ইয়াকিন সেভিংস একাউন্টে কি হারে মুনাফা দেওয়া হয়?',
                'semantic_type': 'islamic_interest_terminology'
            },
            {
                'query': 'ইসলামিক ব্যাংকিং এর বৈশিষ্ট্য কি?',
                'expected_collection': 'faq_yaqeen',
                'expected_question': 'ইয়াকিন একাউন্ট কী?',
                'semantic_type': 'islamic_banking_concept'
            },
            {
                'query': 'শরিয়াহ সম্মত একাউন্ট কোনটি?',
                'expected_collection': 'faq_yaqeen', 
                'expected_question': 'ইয়াকিন একাউন্ট কী?',
                'semantic_type': 'sharia_compliant'
            },
            
            # Retail Banking - Conventional vs Islamic disambiguation
            {
                'query': 'এমটিবি রেগুলার একাউন্টের সুদের হার কত?',
                'expected_collection': 'faq_retail',
                'expected_question': 'এমটিবি রেগুলার সেভিংস একাউন্ট সম্পর্কে জানতে চাই?',
                'semantic_type': 'conventional_banking'
            },
            {
                'query': 'লাখপতি স্কিম কি?',
                'expected_collection': 'faq_retail',
                'expected_question': 'এমটিবি লাখপতি সেভিংস স্কিম সম্পর্কে জানতে চাই?',
                'semantic_type': 'savings_scheme'
            },
            
            # SME Banking
            {
                'query': 'ব্যবসায়িক ঋণের সুদের হার কত?',
                'expected_collection': 'faq_sme',
                'expected_question': 'এসএমই একাউন্ট বা লোন এর ইন্টারেস্ট রেট কত?',
                'semantic_type': 'business_loan'
            },
            {
                'query': 'উদ্যোক্তা একাউন্ট খুলতে কি লাগে?',
                'expected_collection': 'faq_sme',
                'expected_question': 'এসএমই একাউন্ট খুলতে কি কি কাগজ লাগে?',
                'semantic_type': 'entrepreneur_account'
            },
            
            # Card Banking
            {
                'query': 'ক্রেডিট কার্ডের বার্ষিক ফি কত?',
                'expected_collection': 'faq_card',
                'expected_question': 'ক্রেডিট কার্ডের বার্ষিক ফি কত?',
                'semantic_type': 'card_fees'
            },
            {
                'query': 'ডেবিট কার্ড ব্লক করতে চাই',
                'expected_collection': 'faq_card',
                'expected_question': 'কার্ড ব্লক/আনব্লক করার নিয়ম কি?',
                'semantic_type': 'card_services'
            },
            
            # Women Banking
            {
                'query': 'মহিলাদের জন্য বিশেষ একাউন্ট আছে কি?',
                'expected_collection': 'faq_women',
                'expected_question': 'অঙ্গনা একাউন্ট সম্পর্কে জানতে চাই?',
                'semantic_type': 'women_specific'
            },
            {
                'query': 'নারী গ্রাহকদের সুবিধা কি?',
                'expected_collection': 'faq_women',
                'expected_question': 'অঙ্গনা একাউন্ট এর সুবিধাসমূহ কি কি?',
                'semantic_type': 'women_benefits'
            },
            
            # Cross-collection confusion tests
            {
                'query': 'অনলাইনে একাউন্ট খোলা যায় কি?',
                'expected_collection': 'faq_payroll',  # This appears in payroll FAQ
                'expected_question': 'অনলাইন ই কি একাউন্ট ওপেন করা যায়?',
                'semantic_type': 'online_account_opening'
            },
            {
                'query': 'একাউন্ট খুলতে শাখায় যেতে হবে কি?',
                'expected_collection': 'faq_payroll',  # Related to online account opening
                'expected_question': 'অনলাইন ই কি একাউন্ট ওপেন করা যায়?',
                'semantic_type': 'branch_vs_online'
            },
            
            # Semantic challenge cases - should fail or go to wrong collection
            {
                'query': 'আমার একাউন্ট নম্বর ভুলে গেছি',
                'expected_collection': 'none',  # Should not match
                'expected_question': '',
                'semantic_type': 'out_of_scope'
            },
            {
                'query': 'আজকের আবহাওয়া কেমন?',
                'expected_collection': 'none',  # Should not match
                'expected_question': '',
                'semantic_type': 'completely_irrelevant'
            }
        ]
        
        return test_cases
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite and populate confusion matrix"""
        test_cases = self.create_test_cases()
        
        print(f"🧪 Running comprehensive confusion matrix analysis...")
        print(f"📊 Total test cases: {len(test_cases)}")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            expected_collection = test_case['expected_collection']
            expected_question = test_case['expected_question']
            semantic_type = test_case['semantic_type']
            
            print(f"🔍 Test {i}/{len(test_cases)}: {semantic_type}")
            print(f"   Query: {query[:50]}...")
            
            # Process query with error handling
            try:
                result = faq_service.answer_query(query, debug=False)
                
                predicted_collection = result.get('collection', 'none') if result['found'] else 'none'
                predicted_question = result.get('matched_question', '') if result['found'] else ''
                confidence = result.get('confidence', 0.0)
                found = result['found']
                
            except Exception as e:
                print(f"   ❌ ERROR processing query: {e}")
                predicted_collection = 'error'
                predicted_question = ''
                confidence = 0.0
                found = False
                result = {'found': False, 'confidence': 0.0}
            
            # Add to confusion matrix
            self.add_result(
                query=query,
                expected_collection=expected_collection,
                predicted_collection=predicted_collection,
                expected_question=expected_question,
                predicted_question=predicted_question,
                confidence=confidence,
                found=found,
                semantic_type=semantic_type
            )
            
            # Show result
            if found:
                correct = expected_collection == predicted_collection
                status = "✅ CORRECT" if correct else "❌ WRONG COLLECTION"
                print(f"   Result: {status} - {predicted_collection} ({confidence:.1%})")
            else:
                status = "✅ CORRECT" if expected_collection == 'none' else "❌ MISSED"
                print(f"   Result: {status} - No match ({confidence:.1%})")
            
            print()
    
    def generate_confusion_matrix(self) -> pd.DataFrame:
        """Generate collection-level confusion matrix"""
        collections = list(self.collection_mapping.keys()) + ['none']
        matrix = pd.DataFrame(0, index=collections, columns=collections)
        
        for result in self.results:
            expected = result['expected_collection']
            predicted = result['predicted_collection']
            matrix.loc[expected, predicted] += 1
            
        return matrix
    
    def generate_semantic_type_analysis(self) -> pd.DataFrame:
        """Analyze performance by semantic type"""
        semantic_analysis = []
        
        semantic_types = list(set(r['semantic_type'] for r in self.results))
        
        for sem_type in semantic_types:
            type_results = [r for r in self.results if r['semantic_type'] == sem_type]
            
            total = len(type_results)
            correct = sum(1 for r in type_results if r['collection_match'])
            found = sum(1 for r in type_results if r['found'])
            avg_confidence = np.mean([r['confidence'] for r in type_results])
            
            semantic_analysis.append({
                'semantic_type': sem_type,
                'total_tests': total,
                'correct_collection': correct,
                'found_matches': found,
                'collection_accuracy': correct / total if total > 0 else 0,
                'match_rate': found / total if total > 0 else 0,
                'avg_confidence': avg_confidence
            })
        
        return pd.DataFrame(semantic_analysis)
    
    def plot_confusion_matrix(self, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix visualization"""
        matrix = self.generate_confusion_matrix()
        
        # Create readable labels
        readable_labels = [self.collection_mapping.get(col, col.title()) for col in matrix.columns]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=readable_labels,
                   yticklabels=readable_labels)
        
        plt.title('Bengali FAQ System - Collection Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Collection', fontsize=12)
        plt.ylabel('Expected Collection', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
        
        return matrix
    
    def plot_semantic_analysis(self, save_path: str = "semantic_analysis.png"):
        """Plot semantic type analysis"""
        semantic_df = self.generate_semantic_type_analysis()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Collection accuracy by semantic type
        semantic_df_sorted = semantic_df.sort_values('collection_accuracy', ascending=True)
        bars1 = ax1.barh(semantic_df_sorted['semantic_type'], semantic_df_sorted['collection_accuracy'])
        ax1.set_xlabel('Collection Accuracy')
        ax1.set_title('Collection Accuracy by Semantic Type')
        ax1.set_xlim(0, 1)
        
        # Add percentage labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', ha='left', va='center')
        
        # Average confidence by semantic type
        bars2 = ax2.barh(semantic_df_sorted['semantic_type'], semantic_df_sorted['avg_confidence'])
        ax2.set_xlabel('Average Confidence')
        ax2.set_title('Average Confidence by Semantic Type')
        ax2.set_xlim(0, 1)
        
        # Add confidence labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Semantic analysis saved to: {save_path}")
        
        return semantic_df
    
    def generate_detailed_report(self, save_path: str = "confusion_matrix_report.json"):
        """Generate detailed JSON report"""
        matrix = self.generate_confusion_matrix()
        semantic_df = self.generate_semantic_type_analysis()
        
        # Calculate overall metrics
        total_tests = len(self.results)
        correct_collections = sum(1 for r in self.results if r['collection_match'])
        found_matches = sum(1 for r in self.results if r['found'])
        avg_confidence = np.mean([r['confidence'] for r in self.results])
        
        report = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_test_cases': total_tests,
                'system_version': 'Enhanced Semantic v2.0'
            },
            'overall_metrics': {
                'collection_accuracy': correct_collections / total_tests,
                'match_rate': found_matches / total_tests,
                'average_confidence': avg_confidence,
                'total_correct': correct_collections,
                'total_found': found_matches,
                'total_missed': total_tests - found_matches
            },
            'confusion_matrix': matrix.to_dict(),
            'semantic_type_analysis': semantic_df.to_dict('records'),
            'detailed_results': self.results
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 Detailed report saved to: {save_path}")
        return report
    
    def print_summary(self):
        """Print comprehensive summary"""
        total_tests = len(self.results)
        correct_collections = sum(1 for r in self.results if r['collection_match'])
        found_matches = sum(1 for r in self.results if r['found'])
        avg_confidence = np.mean([r['confidence'] for r in self.results])
        
        print("=" * 60)
        print("📊 CONFUSION MATRIX ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"🧪 Total test cases: {total_tests}")
        print(f"✅ Correct collections: {correct_collections} ({correct_collections/total_tests:.1%})")
        print(f"🎯 Found matches: {found_matches} ({found_matches/total_tests:.1%})")
        print(f"📈 Average confidence: {avg_confidence:.1%}")
        print(f"❌ Missed queries: {total_tests - found_matches}")
        print(f"🔄 Wrong collections: {found_matches - correct_collections}")
        print("=" * 60)
        
        # Best and worst performing semantic types
        semantic_df = self.generate_semantic_type_analysis()
        best_type = semantic_df.loc[semantic_df['collection_accuracy'].idxmax()]
        worst_type = semantic_df.loc[semantic_df['collection_accuracy'].idxmin()]
        
        print(f"🏆 Best semantic type: {best_type['semantic_type']} ({best_type['collection_accuracy']:.1%})")
        print(f"🔍 Worst semantic type: {worst_type['semantic_type']} ({worst_type['collection_accuracy']:.1%})")
        print("=" * 60)

def main():
    """Run comprehensive confusion matrix analysis"""
    # Check if FAQ service is ready
    if not faq_service.initialized:
        print("❌ FAQ Service not initialized!")
        return
    
    print("🚀 Bengali FAQ System - Confusion Matrix Analysis")
    print("=" * 60)
    
    # Create analyzer
    analyzer = FAQConfusionMatrix()
    
    # Run comprehensive test
    analyzer.run_comprehensive_test()
    
    # Generate all analysis
    print("📊 Generating confusion matrix visualization...")
    matrix = analyzer.plot_confusion_matrix("faq_confusion_matrix.png")
    
    print("📈 Generating semantic type analysis...")
    semantic_df = analyzer.plot_semantic_analysis("faq_semantic_analysis.png")
    
    print("📋 Generating detailed report...")
    report = analyzer.generate_detailed_report("faq_confusion_report.json")
    
    # Print summary
    analyzer.print_summary()
    
    print("✅ Confusion matrix analysis completed!")
    print("📁 Generated files:")
    print("   - faq_confusion_matrix.png")
    print("   - faq_semantic_analysis.png") 
    print("   - faq_confusion_report.json")

if __name__ == "__main__":
    main()