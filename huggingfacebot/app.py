import os
from flask import Flask, render_template, request, jsonify
import spacy
import pytextrank
from keybert import KeyBERT
import google.generativeai as genai
import requests

HF_TOKEN = "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"
GENERATIVE_API_KEY = "AIzaSyAMyRg4X2plF4Lk2vsngRYBcKnNzW4xd7g"
genai.configure(api_key=GENERATIVE_API_KEY)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')

app = Flask(__name__)

def call_gemini_model(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    print(response.text)
    return response.text

def extract_keywords(text, top_n=2):
    keywords = keybert_model.extract_keywords(text)
    return [keyword[0] for keyword in keywords]

def fetch_huggingface_models(search=None, limit=10):
    url = "https://huggingface.co/api/models"
    params = {
        "search": search,
        "limit": limit,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        models = response.json()
        return models
    else:
        return f"Error: {response.status_code}"

def search_models_based_on_query(query, limit=10):
    keywords = extract_keywords(query)
    results = []
    for keyword in keywords:
        models = fetch_huggingface_models(search=keyword, limit=limit)
        if isinstance(models, list):
            for model in models:
                model_info = {
                    'model_id': model['modelId'],
                    'description': model.get('description', 'No description available'),
                    'downloads': model.get('downloads', 'N/A')
                }
                results.append(model_info)
        else:
            results.append({'error': models})
    return results

def extract_context_from_models(models):
    model_names = [model['model_id'] for model in models]
    return " ".join(model_names) if model_names else "No context available."

def format_gemini_response(text):
    text = text.replace("**", "")
    
    formatted_lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    formatted_text = "<br>".join(formatted_lines)
    
    return formatted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
@app.route('/api/data', methods=['POST'])
def handle_query():
    user_message = request.json.get('message')
    
    model_search_results = search_models_based_on_query(user_message, limit=10)
    
    if model_search_results:
        context = extract_context_from_models(model_search_results)
        prompt = f"User question is: {user_message}. Info about the question is: {context}. Now ans user query according to info provided to you(use only which is relevant and discard all other irrelevant models). Just provide ans in list,numbering the ans, no need for explanation.No need for repeating question or any recommendation or notes"
        answer = call_gemini_model(prompt)
        
        formatted_answer = format_gemini_response(answer)
        print(formatted_answer)
        
        return render_template('response.html', response=formatted_answer)
    else:
        return jsonify({"response": "No relevant models found."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

