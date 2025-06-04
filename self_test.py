from faq_service import faq_service
import os
import glob
import json
import re

def parse_questions(filepath):
    questions=[]
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line.startswith('Question:'):
                questions.append(line[len('Question:'):].strip())
            elif line.startswith('প্রশ্ন:'):
                questions.append(line[len('প্রশ্ন:'):].strip())
    return questions


def main():
    files=glob.glob(os.path.join('faq_data','*.txt'))
    total=0
    correct=0
    results=[]
    limit = 20
    for fp in files:
        qs = parse_questions(fp)
        for q in qs:
            if total >= limit:
                break
            total += 1
            res = faq_service.answer_query(q)
            results.append({
                'query': q,
                'found': res.get('found'),
                'matched_question': res.get('matched_question'),
                'score': res.get('confidence', 0.0),
                'source': res.get('source')
            })
            if res.get('found') and res.get('matched_question') == q:
                correct += 1
        if total >= limit:
            break
    accuracy=correct/total if total else 0.0
    print(f'Accuracy: {accuracy:.1%} ({correct}/{total})')
    with open('self_test_results.json','w',encoding='utf-8') as f:
        json.dump({'accuracy':accuracy,'total':total,'correct':correct,'results':results},f,ensure_ascii=False,indent=2)

if __name__=='__main__':
    main()
