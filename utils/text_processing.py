import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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

def jaccard_similarity(query, doc):
    query_set = set(query.split())
    doc_set = set(doc.split())
    if not query_set or not doc_set:
        return 0.0
    return len(query_set & doc_set) / len(query_set | doc_set)

def bm25_similarity(query, documents):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    return bm25.get_scores(tokenized_query)

def lsa_similarity(query, documents, vectorizer, tfidf_matrix, n_components=100):
    # Reduksi dimensi dengan LSA
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    # Transformasi kueri ke ruang LSA
    query_vector = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vector)
    # Hitung cosine similarity
    lsa_scores = cosine_similarity(query_lsa, lsa_matrix)[0]
    return lsa_scores