from flask import Flask, request, jsonify
import json
import os
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# === PREPROCESSING SETUP ===
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def highlight_lyric(lyric, query):
    query = query.lower()
    # Ambil kata-kata kunci dari query
    query_words = set(word_tokenize(query))
    # Buat regex pattern untuk mencocokkan kata-kata tersebut dalam lyric
    def replacer(match):
        word = match.group(0)
        return f"**{word}**" if word.lower() in query_words else word
    pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
    return pattern.sub(replacer, lyric)

# === LOAD DATASET & VECTORIZE ===
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, 'datasets', 'clean_dataset.json')

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

documents = []
original_data = []

for item in dataset:
    if 'preprocessed_lyric' in item and item['preprocessed_lyric']:
        documents.append(item['preprocessed_lyric'])
        original_data.append(item)

# vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

tfidf_matrix = vectorizer.fit_transform(documents)

# === API ROUTE ===
@app.route('/search', methods=['POST'])
def search_post():
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({"error": "JSON body must include 'query'"}), 400

    query = data['query']
    count = data.get('count', 5)

    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed])
    cos_sim = cosine_similarity(query_vector, tfidf_matrix)[0]

    results = [(i, score) for i, score in enumerate(cos_sim) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return jsonify({"message": "No matching lyrics found."}), 404

    top_matches = []
    for i, (index, score) in enumerate(results[:count]):
        match = original_data[index]
        lyric = match.get('lyric', '')
        highlighted = highlight_lyric(lyric, query)
        top_matches.append({
            "title": match.get('title'),
            "artist": match.get('artist'),
            "score": round(score, 4),
            "lyric": highlighted
        })

    return jsonify({"results": top_matches})

# === RUN FLASK APP ===
if __name__ == '__main__':
    app.run(debug=True)
