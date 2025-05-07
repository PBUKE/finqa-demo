import streamlit as st
from models.gptneo_runner import gptneo_infer
from models.tinyllama_runner import tinyllama_infer
from models.mistral_runner import mistral_infer
from utils.ensemble import ensemble_vote
from utils.evaluation import normalize_answer
from utils.parser import extract_context

st.title("📊 Financial QA with Multiple LLMs")

uploaded_file = st.file_uploader("Upload financial document (JSON)", type=["json"])
question = st.text_input("Enter your financial question")
model_choice = st.selectbox("Choose model(s)", ["GPT-Neo", "TinyLLaMA", "Mistral", "All"])

if uploaded_file and question:
    context = extract_context(uploaded_file)
    answers = {}

    if model_choice in ["GPT-Neo", "All"]:
        answers['GPT-Neo'] = gptneo_infer(question, context)
    if model_choice in ["TinyLLaMA", "All"]:
        answers['TinyLLaMA'] = tinyllama_infer(question, context)
    if model_choice in ["Mistral", "All"]:
        answers['Mistral'] = mistral_infer(question, context)

    st.subheader("🧠 Answers")
    for model, ans in answers.items():
        st.write(f"**{model}:** {ans}")

    if model_choice == "All":
        final_answer = ensemble_vote(list(answers.values()))
        st.success(f"✅ Final Answer (ensemble): {final_answer}")