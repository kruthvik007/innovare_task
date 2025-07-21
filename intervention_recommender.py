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

    with open("FOT-Toolkit_pages_4_258_ocr.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = re.split(r"\n\s*(Strategy|Intervention|PLAN)\s*[:\-]?", full_text, flags=re.IGNORECASE)

    clean_chunks = []
    for i in range(1, len(chunks), 2):
        title = chunks[i].strip()
        body = chunks[i+1].strip()
        clean_chunks.append({"title": title, "text": body})

    strategy_texts = [chunk['text'] for chunk in clean_chunks]
    strategy_embeddings = embed_model.encode(strategy_texts, convert_to_tensor=True)

    index = faiss.IndexFlatL2(strategy_embeddings.shape[1])
    index.add(strategy_embeddings.cpu().numpy())

    def find_top_k_strategies(question: str, narrative: str, k=5):
        query = f"Q: {question}\nContext: {narrative}"
        query_embedding = embed_model.encode([query])
        D, I = index.search(np.array(query_embedding), k)
        return [clean_chunks[i] for i in I[0]]

    retrieved_strategies_per_question = [
        find_top_k_strategies(q, student_narrative, k=5) for q in top_questions
    ]

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    final_answers = []

    for i, question in enumerate(top_questions):
        prompt = f"""
        You are a school intervention expert tasked with recommending the most appropriate intervention strategy for a student.
        Student Narrative:
        {student_narrative}

        Planning Question:
        {question}

        Below are several candidate intervention strategies. Read all of them carefully.

        Your task:
        - Choose the **single most relevant strategy** based on the planning question and student narrative.
        - Ground your answer in the selected strategy, but do **not copy it verbatim**.
        - Focus on **actionable insights** that directly relate to the student's situation.
        - Do **not** include any extra explanation or strategy names in your output.

        Strategies:
        """

        for j, strat in enumerate(retrieved_strategies_per_question[i]):
            strategy_text = strat['text'] if isinstance(strat, dict) else str(strat)
            prompt += f"{j+1}. {strategy_text.strip()}\n\n"

        prompt += "\nReturn only a clear and concise natural language answer grounded in one strategy."

        response = llama_pipeline(
            textwrap.dedent(prompt),
            max_new_tokens=50,
            do_sample=False,
            return_full_text=False,
            repetition_penalty=1.2,
        )[0]["generated_text"].strip()
        
        final_answers.append((question, response.strip()))

    for i, (question, answer) in enumerate(final_answers, 1):
        print(f"Q{i}: {question}\n Answer: {answer}\n{'-'*80}")

    json_data = [{"question": q, "answer": a} for q, a in final_answers]
    json_str = json.dumps(json_data, indent=2)
    print(json_str)
    
    return json_str
