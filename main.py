from flask import Flask, request, jsonify
from pydantic import BaseModel
import json
import os
import time
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import preprocess_text, jaccard_similarity, bm25_similarity, lsa_similarity, word_embeddings_similarity, bert_similarity
from flask_cors import CORS
from typing import Optional

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": ["http://localhost:3000", "*"]}})

# === LOAD DATASET ===
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, 'datasets', 'clean_dataset.json')

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

documents = []
original_data = []
raw_documents = []  # Store raw lyrics for BERT

for item in dataset:
    if 'preprocessed_lyric' in item and item['preprocessed_lyric']:
        documents.append(item['preprocessed_lyric'])
        raw_documents.append(item.get('lyric', ''))  # Store raw lyric
        original_data.append(item)

# === VECTORIZE (TF-IDF untuk TF-IDF dan LSA) ===
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = vectorizer.fit_transform(documents)

# === MODEL INPUT ===
class SearchRequest(BaseModel):
    query: str
    count: Optional[int] = 5

# === HIGHLIGHT LYRICS ===
def highlight_lyric(lyric, query):
    query = query.lower()
    query_words = set(word_tokenize(query))
    def replacer(match):
        word = match.group(0)
        return f"**{word}**" if word.lower() in query_words else word
    pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
    return pattern.sub(replacer, lyric)

# === SEARCH FUNCTION ===
def search_lyrics(query, count):
    query_processed = preprocess_text(query)
    query_processed_we = preprocess_text(query, use_stemming=False)
    query_raw = query  # Raw query for BERT

    # TF-IDF + Cosine Similarity
    # start_time = time.time()
    # query_vector = vectorizer.transform([query_processed])
    # cos_sim = cosine_similarity(query_vector, tfidf_matrix)[0]
    # tfidf_results = [(i, score) for i, score in enumerate(cos_sim) if score > 0]
    # tfidf_results.sort(key=lambda x: x[1], reverse=True)
    # tfidf_time = time.time() - start_time

    # # Jaccard Similarity
    # start_time = time.time()
    # jaccard_scores = [jaccard_similarity(query_processed, doc) for doc in documents]
    # jaccard_results = [(i, score) for i, score in enumerate(jaccard_scores) if score > 0]
    # jaccard_results.sort(key=lambda x: x[1], reverse=True)
    # jaccard_time = time.time() - start_time

    # # BM25
    # start_time = time.time()
    # bm25_scores = bm25_similarity(query_processed, documents)
    # bm25_results = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
    # bm25_results.sort(key=lambda x: x[1], reverse=True)
    # bm25_time = time.time() - start_time

    # # LSA
    # start_time = time.time()
    # lsa_scores = lsa_similarity(query_processed, documents, vectorizer, tfidf_matrix)
    # lsa_results = [(i, score) for i, score in enumerate(lsa_scores) if score > 0]
    # lsa_results.sort(key=lambda x: x[1], reverse=True)
    # lsa_time = time.time() - start_time

    # # Word Embeddings
    # start_time = time.time()
    # we_scores = word_embeddings_similarity(query_processed_we, documents, cache_file=os.path.join(BASE_DIR, 'models', 'doc_embeddings.pkl'))
    # we_results = [(i, score) for i, score in enumerate(we_scores) if score > 0]
    # we_results.sort(key=lambda x: x[1], reverse=True)
    # we_time = time.time() - start_time

    # BERT
    start_time = time.time()
    bert_scores = bert_similarity(query_raw, raw_documents, cache_file=os.path.join(BASE_DIR, 'models', 'bert_doc_embeddings.pkl'))
    bert_results = [(i, score) for i, score in enumerate(bert_scores) if score > 0]
    bert_results.sort(key=lambda x: x[1], reverse=True)
    bert_time = time.time() - start_time

    # Format hasil untuk setiap metode
    def format_results(results, method_name, original_data, count):
        method_results = []
        for i, (index, score) in enumerate(results[:count]):
            match = original_data[index]
            lyric = match.get('lyric', '')
            highlighted = highlight_lyric(lyric, query)
            method_results.append({
                "title": match.get('title'),
                "artist": match.get('artist'),
                "score": round(score * 100, 2),  # Convert to percentage
                "lyric": highlighted
            })
        return method_results

    response = {
        # "tfidf": {
        #     "results": format_results(tfidf_results, "TF-IDF + Cosine Similarity", original_data, count),
        #     "execution_time": round(tfidf_time, 4),
        #     "matches_found": len(tfidf_results)
        # },
        # "jaccard": {
        #     "results": format_results(jaccard_results, "Jaccard Similarity", original_data, count),
        #     "execution_time": round(jaccard_time, 4),
        #     "matches_found": len(jaccard_results)
        # },
        # "bm25": {
        #     "results": format_results(bm25_results, "BM25", original_data, count),
        #     "execution_time": round(bm25_time, 4),
        #     "matches_found": len(bm25_results)
        # },
        # "lsa": {
        #     "results": format_results(lsa_results, "LSA", original_data, count),
        #     "execution_time": round(lsa_time, 4),
        #     "matches_found": len(lsa_results)
        # },
        # "word_embeddings": {
        #     "results": format_results(we_results, "Word Embeddings", original_data, count),
        #     "execution_time": round(we_time, 4),
        #     "matches_found": len(we_results)
        # },
        "bert": {
            "results": format_results(bert_results, "BERT", original_data, count),
            "execution_time": round(bert_time, 4),
            "matches_found": len(bert_results)
        }
    }

    return response

# === API ROUTE ===
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query cannot be empty"}), 400

        search_request = SearchRequest(**data)

        response = search_lyrics(search_request.query, search_request.count)

        if not any(response[method]["results"] for method in response):
            return jsonify({"error": "No matching lyrics found"}), 404

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": "Invalid request data"}), 400

# === RUN FLASK APP ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)