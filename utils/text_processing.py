import re
import os
import pickle
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# === PREPROCESSING SETUP ===
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

def preprocess_text(text, use_stemming=True):
    text = text.lower()
    # Hapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi
    tokens = word_tokenize(text)
    # Hapus stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # Stemming (opsional)
    if use_stemming:
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

def word_embeddings_similarity(query, documents, cache_file='models/doc_embeddings.pkl'):
    # Gunakan model sentence-transformers multibahasa
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models')
    # Cek apakah embeddings sudah di-cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        doc_embeddings = model.encode(documents)
        with open(cache_file, 'wb') as f:
            pickle.dump(doc_embeddings, f)
    query_embedding = model.encode([query])[0]
    we_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    return we_scores

def bert_similarity(query, documents, cache_file='models-bert/bert_doc_embeddings.pkl'):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased', cache_dir='models-bert')
    model = AutoModel.from_pretrained('distilbert-base-multilingual-cased', cache_dir='models-bert')
    model.eval()  # Set to evaluation mode

    # Function to get BERT embeddings
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding (first token) or mean pooling
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
        return embedding

    # Check if document embeddings are cached
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        # Generate embeddings for all documents
        doc_embeddings = [get_bert_embedding(doc) for doc in documents]
        with open(cache_file, 'wb') as f:
            pickle.dump(doc_embeddings, f)

    # Generate embedding for query
    query_embedding = get_bert_embedding(query)

    # Compute cosine similarity
    bert_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    return bert_scores