def ensemble_vote(answers):
    from collections import Counter
    common = Counter(answers).most_common(1)
    return common[0][0] if common else "No consensus"

# utils/evaluation.py
def normalize_answer(ans):
    return str(ans).strip().lower().replace('%', '').replace('$', '')

# utils/parser.py
import json
def extract_context(uploaded_file):
    json_data = json.load(uploaded_file)
    return json.dumps(json_data)[:500] 