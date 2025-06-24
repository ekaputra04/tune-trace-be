import json
import os
from utils.text_processing import preprocess_text, jaccard_similarity, bm25_similarity, lsa_similarity, word_embeddings_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, 'datasets', 'clean_dataset.json')
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

documents = [item['preprocessed_lyric'] for item in dataset if 'preprocessed_lyric' in item and item['preprocessed_lyric']]
original_data = [item for item in dataset if 'preprocessed_lyric' in item and item['preprocessed_lyric']]

# Inisialisasi TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = vectorizer.fit_transform(documents)

# Fungsi untuk menjalankan semua metode
def run_methods(query):
    query_processed = preprocess_text(query)
    query_processed_we = preprocess_text(query, use_stemming=False)

    # TF-IDF
    query_vector = vectorizer.transform([query_processed])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    tfidf_results = [(i, score) for i, score in enumerate(tfidf_scores) if score > 0]
    tfidf_results.sort(key=lambda x: x[1], reverse=True)

    # Jaccard
    jaccard_scores = [jaccard_similarity(query_processed, doc) for doc in documents]
    jaccard_results = [(i, score) for i, score in enumerate(jaccard_scores) if score > 0]
    jaccard_results.sort(key=lambda x: x[1], reverse=True)

    # BM25
    bm25_scores = bm25_similarity(query_processed, documents)
    bm25_results = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
    bm25_results.sort(key=lambda x: x[1], reverse=True)

    # LSA
    lsa_scores = lsa_similarity(query_processed, documents, vectorizer, tfidf_matrix)
    lsa_results = [(i, score) for i, score in enumerate(lsa_scores) if score > 0]
    lsa_results.sort(key=lambda x: x[1], reverse=True)

    # Word Embedding
    we_scores = word_embeddings_similarity(query_processed_we, documents, cache_file=os.path.join(BASE_DIR, 'models', 'doc_embeddings.pkl'))
    we_results = [(i, score) for i, score in enumerate(we_scores) if score > 0]
    we_results.sort(key=lambda x: x[1], reverse=True)

    # Format hasil
    def format_top_result(results):
        if results:
            idx, score = results[0]
            return {
                "title": original_data[idx].get('title'),
                "artist": original_data[idx].get('artist'),
                "score": round(score * 100, 2),
                "lyric": original_data[idx].get('lyric'),
            }
        return None

    return {
        "tfidf": format_top_result(tfidf_results),
        "jaccard": format_top_result(jaccard_results),
        "bm25": format_top_result(bm25_results),
        "lsa": format_top_result(lsa_results),
        "word_embedding": format_top_result(we_results)
    }

# Uji dengan kueri sinonim atau konteks terkait
test_queries = [
    "cinta yang hilang",
    "kasih yang pergi",
    "hati yang patah"
]

for query in test_queries:
    print(f"\nKueri: {query}")
    results = run_methods(query)
    for method, result in results.items():
        print(f"{method}: {result}")