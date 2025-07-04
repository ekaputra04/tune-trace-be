from flask import Flask, request, jsonify
from pydantic import BaseModel
import json
import os
import time
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import preprocess_text, jaccard_similarity, bm25_similarity, lsa_similarity, word_embeddings_similarity
from flask_cors import CORS
from typing import Optional

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": ["http://localhost:3000", "*"]}, r"/context-analysis": {"origins": ["http://localhost:3000", "*"]}})

# === LOAD DATASET ===
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, 'datasets', 'clean_dataset.json')

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

documents = []
original_data = []
raw_documents = []

for item in dataset:
    if 'preprocessed_lyric' in item and item['preprocessed_lyric']:
        documents.append(item['preprocessed_lyric'])
        raw_documents.append(item.get('lyric', ''))
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

    # TF-IDF + Cosine Similarity
    start_time = time.time()
    query_vector = vectorizer.transform([query_processed])
    cos_sim = cosine_similarity(query_vector, tfidf_matrix)[0]
    tfidf_results = [(i, score) for i, score in enumerate(cos_sim) if score > 0]
    tfidf_results.sort(key=lambda x: x[1], reverse=True)
    tfidf_time = time.time() - start_time

    # Jaccard Similarity
    start_time = time.time()
    jaccard_scores = [jaccard_similarity(query_processed, doc) for doc in documents]
    jaccard_results = [(i, score) for i, score in enumerate(jaccard_scores) if score > 0]
    jaccard_results.sort(key=lambda x: x[1], reverse=True)
    jaccard_time = time.time() - start_time

    # BM25
    start_time = time.time()
    bm25_scores = bm25_similarity(query_processed, documents)
    bm25_results = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
    bm25_results.sort(key=lambda x: x[1], reverse=True)
    bm25_time = time.time() - start_time

    # LSA
    start_time = time.time()
    lsa_scores = lsa_similarity(query_processed, documents, vectorizer, tfidf_matrix)
    lsa_results = [(i, score) for i, score in enumerate(lsa_scores) if score > 0]
    lsa_results.sort(key=lambda x: x[1], reverse=True)
    lsa_time = time.time() - start_time

    # Word Embeddings
    start_time = time.time()
    we_scores = word_embeddings_similarity(query_processed_we, documents, cache_file=os.path.join(BASE_DIR, 'models', 'doc_embeddings.pkl'))
    we_results = [(i, score) for i, score in enumerate(we_scores) if score > 0]
    we_results.sort(key=lambda x: x[1], reverse=True)
    we_time = time.time() - start_time

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
                "score": round(score * 100, 2),
                "lyric": highlighted
            })
        return method_results

    response = {
        "tfidf": {
            "results": format_results(tfidf_results, "TF-IDF + Cosine Similarity", original_data, count),
            "execution_time": round(tfidf_time, 4),
            "matches_found": len(tfidf_results)
        },
        "jaccard": {
            "results": format_results(jaccard_results, "Jaccard Similarity", original_data, count),
            "execution_time": round(jaccard_time, 4),
            "matches_found": len(jaccard_results)
        },
        "bm25": {
            "results": format_results(bm25_results, "BM25", original_data, count),
            "execution_time": round(bm25_time, 4),
            "matches_found": len(bm25_results)
        },
        "lsa": {
            "results": format_results(lsa_results, "LSA", original_data, count),
            "execution_time": round(lsa_time, 4),
            "matches_found": len(lsa_results)
        },
        "word_embeddings": {
            "results": format_results(we_results, "Word Embeddings", original_data, count),
            "execution_time": round(we_time, 4),
            "matches_found": len(we_results)
        }
    }

    return response

# === NEW CONTEXT ANALYSIS ENDPOINT ===
@app.route('/context-analysis', methods=['POST'])
def context_analysis():
    try:
        data = request.get_json()
        if not data or 'queries' not in data or not isinstance(data['queries'], list):
            return jsonify({"error": "Queries must be a non-empty list"}), 400

        results = []
        for query in data['queries']:
            response = search_lyrics(query, count=1)  # Ambil 1 hasil per metode untuk analisis
            context_result = {
                "query": query,
                "tfidf": response["tfidf"]["results"][0] if response["tfidf"]["results"] else None,
                "jaccard": response["jaccard"]["results"][0] if response["jaccard"]["results"] else None,
                "bm25": response["bm25"]["results"][0] if response["bm25"]["results"] else None,
                "lsa": response["lsa"]["results"][0] if response["lsa"]["results"] else None,
                "word_embeddings": response["word_embeddings"]["results"][0] if response["word_embeddings"]["results"] else None
            }
            results.append(context_result)

        # Analisis konsistensi (contoh sederhana: hitung lagu yang sama di Word Embedding dan LSA)
        consistency = {
            "word_embeddings_lsa_matches": 0,
            "details": []
        }
        reference_songs = set()
        for result in results:
            we_song = result["word_embeddings"]["title"] if result["word_embeddings"] else None
            lsa_song = result["lsa"]["title"] if result["lsa"] else None
            if we_song and lsa_song and we_song == lsa_song:
                consistency["word_embeddings_lsa_matches"] += 1
                consistency["details"].append({
                    "query": result["query"],
                    "song": we_song,
                    "artist": result["word_embeddings"]["artist"]
                })

        return jsonify({
            "context_results": results,
            "consistency_analysis": consistency
        })

    except ValueError as e:
        return jsonify({"error": "Invalid request data"}), 400

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