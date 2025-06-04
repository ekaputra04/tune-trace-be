import json
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import preprocess_text, jaccard_similarity, bm25_similarity, lsa_similarity, word_embeddings_similarity

# === LOAD DATASET ===
BASE_URL = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_URL, 'datasets', 'clean_dataset.json')

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# === GUNAKAN PREPROCESSED LYRIC ===
documents = []
original_data = []

for item in dataset:
    if 'preprocessed_lyric' in item:
        documents.append(item['preprocessed_lyric'])
        original_data.append(item)

# === USER INPUT & PREPROCESS ===
query = input("Masukkan penggalan lirik lagu: ")
count = int(input("Berapa hasil yang ingin ditampilkan: "))
query_processed = preprocess_text(query)  # Untuk TF-IDF, Jaccard, BM25, LSA
query_processed_we = preprocess_text(query, use_stemming=False)  # Untuk Word Embeddings

# === TF-IDF + COSINE SIMILARITY ===
start_time = time.time()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query_processed])
cos_sim = cosine_similarity(query_vector, tfidf_matrix)[0]
tfidf_results = [(i, score) for i, score in enumerate(cos_sim) if score > 0]
tfidf_results.sort(key=lambda x: x[1], reverse=True)
tfidf_time = time.time() - start_time

# === JACCARD SIMILARITY ===
start_time = time.time()
jaccard_scores = [jaccard_similarity(query_processed, doc) for doc in documents]
jaccard_results = [(i, score) for i, score in enumerate(jaccard_scores) if score > 0]
jaccard_results.sort(key=lambda x: x[1], reverse=True)
jaccard_time = time.time() - start_time

# === BM25 SIMILARITY ===
start_time = time.time()
bm25_scores = bm25_similarity(query_processed, documents)
bm25_results = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
bm25_results.sort(key=lambda x: x[1], reverse=True)
bm25_time = time.time() - start_time

# === LSA SIMILARITY ===
start_time = time.time()
lsa_scores = lsa_similarity(query_processed, documents, vectorizer, tfidf_matrix)
lsa_results = [(i, score) for i, score in enumerate(lsa_scores) if score > 0]
lsa_results.sort(key=lambda x: x[1], reverse=True)
lsa_time = time.time() - start_time

# === WORD EMBEDDINGS SIMILARITY ===
start_time = time.time()
we_scores = word_embeddings_similarity(query_processed_we, documents, cache_file=os.path.join(BASE_URL, 'models', 'doc_embeddings.pkl'))
we_results = [(i, score) for i, score in enumerate(we_scores) if score > 0]
we_results.sort(key=lambda x: x[1], reverse=True)
we_time = time.time() - start_time

# === TAMPILKAN HASIL ===
def display_results(results, method_name, original_data, count):
    if results:
        print(f"\n🎵 {method_name} - Ditemukan {len(results)} kemungkinan lagu yang cocok:\n")
        for i, (index, score) in enumerate(results[:count], 1):
            match = original_data[index]
            print(f"{i}. Judul  : {match['title']}")
            print(f"   Artis  : {match['artist']}")
            print(f"   Skor   : {score * 100:.2f}%")
            print(f"   Lirik  : {match['lyric'][:200]}...\n")
    else:
        print(f"\n⚠️ {method_name} - Maaf, tidak ada lirik yang cocok ditemukan.")

# Tampilkan hasil untuk setiap metode
display_results(tfidf_results, "TF-IDF + Cosine Similarity", original_data, count)
print(f"Waktu eksekusi TF-IDF: {tfidf_time:.4f} detik")

display_results(jaccard_results, "Jaccard Similarity", original_data, count)
print(f"Waktu eksekusi Jaccard: {jaccard_time:.4f} detik")

display_results(bm25_results, "BM25", original_data, count)
print(f"Waktu eksekusi BM25: {bm25_time:.4f} detik")

display_results(lsa_results, "LSA", original_data, count)
print(f"Waktu eksekusi LSA: {lsa_time:.4f} detik")

display_results(we_results, "Word Embeddings", original_data, count)
print(f"Waktu eksekusi Word Embeddings: {we_time:.4f} detik")