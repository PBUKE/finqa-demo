def normalize_answer(ans):
    return str(ans).strip().lower().replace('%', '').replace('$', '')