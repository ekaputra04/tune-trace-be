# CHOICE MODEL

# ============== DOWNLOAD MODEL all-MiniLM-L6-v2 ==============

# from transformers import AutoModel, AutoTokenizer
# import os

# BASE_URL = os.path.join(os.path.dirname(__file__), "..")

# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# SAVE_DIR = os.path.join(BASE_URL, "models", "all-MiniLM-L6-v2")

# # Download dan simpan model
# model = AutoModel.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model.save_pretrained(SAVE_DIR)
# tokenizer.save_pretrained(SAVE_DIR)

# print(f"Model disimpan ke: {SAVE_DIR}")

# ============== DOWNLOAD MODEL paraphrase-multilingual-MiniLM-L12-v2 ==============

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model.save_pretrained("models/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer.save_pretrained("models/paraphrase-multilingual-MiniLM-L12-v2")
