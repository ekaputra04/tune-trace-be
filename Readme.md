# 🎵 Tune Trace - Backend

Tune Trace adalah backend API yang digunakan untuk menelusuri lirik lagu berdasarkan input teks dari pengguna. API ini dibangun menggunakan Python (Flask) dan dapat digunakan untuk mendukung aplikasi pencarian lirik secara real-time.

## 🔧 Teknologi

- Python 3.9+
- Flask
- Flask-CORS
- Sastrawi
- NLTK
- Scikit Learn

## 📦 Instalasi

1. **Clone repositori ini**

```bash
git clone https://github.com/username/tune-trace-be.git
cd tune-trace-be
```

2. Buat dan aktifkan virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Menjalankan Server

```bash
python app.py
```

Server akan berjalan di:

```
http://127.0.0.1:5000
```

## 📥 Format Request

Endpoint: `POST /search`

Headers:

```pgsql
Content-Type: application/json
```

Body:

```pgsql
{
  "query": "your_query_here",
  "count": 10
}
```

query: String teks yang ingin dicari dalam database lirik.

count (opsional): Jumlah maksimum hasil yang ingin dikembalikan. Default: 10.

## 📤 Format Response

✅ Jika berhasil:

```json
{
  "results": [
    {
      "lyric": "Hello from the other side",
      "title": "Hello",
      "artist": "Adele"
    },
    ...
  ]
}
```

❌ Jika tidak ditemukan:

```json
{
  "results": []
}
```

❌ Jika terjadi error:

```json
{
  "error": "Internal server error"
}
```

## Dataset

[Song Lyrics Dataset](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset)

[Indonesia 2000an Simple EDA](https://www.kaggle.com/code/muzavan/indonesia-2000an-simple-eda?select=katalagu-indonesia-2000an.json)

### Thankyou
