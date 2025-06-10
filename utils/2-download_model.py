from transformers import AutoTokenizer, AutoModel
import os

# Tentukan direktori untuk menyimpan model di /models (root direktori)
base_dir = os.path.dirname(os.path.dirname(__file__))  # Naik satu level dari /utils ke root
cache_dir = os.path.join(base_dir, 'models-bert')
os.makedirs(cache_dir, exist_ok=True)

# Download tokenizer dan model DistilBERT
model_name = 'distilbert-base-multilingual-cased'
try:
    print(f"Mendownload tokenizer untuk {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Tokenizer berhasil disimpan di {cache_dir}")

    print(f"Mendownload model {model_name}...")
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Model berhasil disimpan di {cache_dir}")

    # Simpan model dan tokenizer untuk penggunaan offline
    tokenizer.save_pretrained(cache_dir)
    model.save_pretrained(cache_dir)
    print("Model dan tokenizer telah disimpan untuk penggunaan offline.")
except Exception as e:
    print(f"Error saat mendownload model: {str(e)}")