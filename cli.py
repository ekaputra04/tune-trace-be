import json
import os
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# === PREPROCESSING SETUP ===
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

def preprocess_text(text):
    text = text.lower()
    # Hapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)    
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi    
    tokens = word_tokenize(text)    
    # Hapus stopwords
    tokens = [word for word in tokens if word not in stopwords]    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]    
    # Gabung kembali
    return ' '.join(tokens)

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

# === TF-IDF VECTORIZATION ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# === USER INPUT & PREPROCESS ===
query = input("Masukkan penggalan lirik lagu: ")
count = int(input("Berapa hasil yang ingin ditampilkan: "))
query_processed = preprocess_text(query)
query_vector = vectorizer.transform([query_processed])

# === COSINE SIMILARITY ===
cos_sim = cosine_similarity(query_vector, tfidf_matrix)[0]

# Ambil semua hasil dengan skor > 0
results = [(i, score) for i, score in enumerate(cos_sim) if score > 0]

# Urutkan berdasarkan skor tertinggi
results.sort(key=lambda x: x[1], reverse=True)

# === TAMPILKAN SEMUA HASIL ===
if results:
    print(f"\n🎵 Ditemukan {len(results)} kemungkinan lagu yang cocok:\n")
    for i, (index, score) in enumerate(results[:count], 1):
        match = original_data[index]
        print(f"{i}. Judul  : {match['title']}")
        print(f"   Artis  : {match['artist']}")
        print(f"   Skor   : {score * 100:.2f}%")
        print(f"   Lirik  : {match['lyric']}...\n")  # tampilkan 200 karakter pertama
else:
    print("\n⚠️ Maaf, tidak ada lirik yang cocok ditemukan.")
