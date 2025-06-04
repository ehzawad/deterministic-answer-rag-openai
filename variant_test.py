import os
import glob
import json
import random
import logging
from faq_service import faq_service

# Configure root logger for more detailed output
logging.getLogger().setLevel(logging.INFO)

synonym_map = {
    'একাউন্ট': ['অ্যাকাউন্ট', 'হিসাব'],
    'অ্যাকাউন্ট': ['একাউন্ট', 'হিসাব'],
    'পাওয়া যায়': ['পেতে পারি'],
    'লাখপতি': ['লক্ষপতি'],
    'কোটিপতি': ['কোটি পতি'],
    'ডেবিট কার্ড': ['ডেবিট কারড', 'কার্ড'],
    'সেভিংস': ['সেভিং'],
    'ইন্সপায়ার': ['ইন্সপায়র'],
    'রেগুলার': ['রেগুলার সেভিংস'],
    'ব্যালেন্স': ['ব্যলেন্স'],
}


def parse_questions(filepath):
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Question:'):
                questions.append(line[len('Question:'):].strip())
            elif line.startswith('প্রশ্ন:'):
                questions.append(line[len('প্রশ্ন:'):].strip())
    return questions


def generate_variant(question):
    variant = question
    for key, options in synonym_map.items():
        if key in variant:
            variant = variant.replace(key, random.choice(options))
    # Randomly remove or add question mark
    if variant.endswith('?') and random.random() < 0.5:
        variant = variant[:-1]
    elif not variant.endswith('?') and random.random() < 0.5:
        variant += '?'
    return variant


def main():
    files = glob.glob(os.path.join('faq_data', '*.txt'))
    total = 0
    correct = 0
    results = []

    # Limit to two questions per file to keep runtime short
    for fp in files:
        qs = parse_questions(fp)[:2]
        for orig_q in qs:
            variant_q = generate_variant(orig_q)
            res = faq_service.answer_query(variant_q, debug=True)
            match = res.get('matched_question')
            score = res.get('confidence', 0.0)
            results.append({
                'variant_query': variant_q,
                'expected_question': orig_q,
                'matched_question': match,
                'score': score,
                'details': res.get('match_details', {})
            })
            total += 1
            if res.get('found') and match == orig_q:
                correct += 1

    accuracy = correct / total if total else 0.0
    print(f'Variant Accuracy: {accuracy:.1%} ({correct}/{total})')
    with open('variant_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({'accuracy': accuracy, 'total': total, 'correct': correct, 'results': results}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
