import json

def extract_context(uploaded_file):
    json_data = json.load(uploaded_file)
    return json.dumps(json_data)[:500]

