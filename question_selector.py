import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import re
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import transformers
from dotenv import load_dotenv
import os

def get_top_questions(student_narrative, top_k = 7):

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    mistral_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    mistral_pipeline = pipeline(
        "text-generation",
        model=mistral_model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open("questions.txt", "r", encoding="utf-8") as f:
        questions = f.read().split('\n')

    question_embeddings = embed_model.encode(questions, show_progress_bar=True)

    faiss.normalize_L2(question_embeddings)
    index = faiss.IndexFlatIP(len(question_embeddings[0]))
    index.add(np.array(question_embeddings))

    query_embedding = embed_model.encode([student_narrative])
    faiss.normalize_L2(query_embedding)

    _, indices = index.search(np.array(query_embedding), top_k)

    top_k_questions = [questions[i] for i in indices[0]]

    print("Top K Matched Questions:\n")
    for rank, q in enumerate(top_k_questions, 1):
        print(f"Q{rank}: {q}\n")

    mistral_prompt = f"""You are a school intervention expert. Your task is to select the 3 most relevant planning questions based on the student's situation.
    Only use the information in the student narrative to choose the questions.

    Student Narrative:
    {student_narrative}

    Planning Questions:
    {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(top_k_questions)])}

    INSTRUCTIONS:
    - Only select 3 questions.
    - Do not answer the questions.
    - Do not explain your choices.
    - Do not add any extra text.
    - Format the output exactly like this:

    Q1: <question>
    Q2: <question>
    Q3: <question>
    """

    response = mistral_pipeline(
        textwrap.dedent(mistral_prompt),
        return_full_text=False,
        num_return_sequences=1,
        temperature=0.2,
        top_k=20,
        top_p=0.8,
        repetition_penalty=1.2,
        max_new_tokens=200,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_text = response[0]["generated_text"]
    
    print(output_text)
        
    top_questions = []
    for line in output_text.splitlines():
        line = line.strip()
        if ": " in line:
            question = line.split(": ", 1)[1]
            top_questions.append(question)
    
    print("Top 3 Questions:")
    for i, q in enumerate(top_questions, 1):
        print(f"Q{i}: {q}")
    
    top_questions_json = {
    f"Q{i+1}": q for i, q in enumerate(top_questions)
}
    
    return json.dumps(top_questions_json)
