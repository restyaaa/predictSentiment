import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import gensim
import nltk

import compress_pickle

# Memuat model FastText dari file .pkl.gz yang terkompresi
model_vec = compress_pickle.load('models/cc.id.300.pkl.gz')

# Muat model klasifikasi
knn = pickle.load(open('models/KNN+GridSearch.pkl', 'rb'))
svm = pickle.load(open('models/SVM+GridSearch.pkl', 'rb'))

# Inisialisasi FastAPI
app = FastAPI()
nltk.download('punkt_tab')
nltk.download('stopwords')
# Setup Jinja2 untuk template HTML
templates = Jinja2Templates(directory="templates")

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = word_tokenize(text)
    
    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    words = [stemmer.stem(word) for word in words]

    return words  # Mengembalikan list kata yang sudah diproses

# Endpoint untuk menampilkan halaman utama (form input)
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint untuk menerima form input dan memberikan prediksi sentimen
@app.post("/predict")
async def predict_sentiment(request: Request, text: str = Form(...), chosen_model: str = Form(...)):
    # Preprocessing teks
    clean_words = preprocess_text(text)
    
    # Ambil vektor untuk setiap kata yang ada dalam model FastText
    word_vectors = [model_vec[word] for word in clean_words if word in model_vec]
    
    # Jika ada kata yang ditemukan dalam model, hitung rata-rata vektornya
    if word_vectors:
        text_vector = np.mean(word_vectors, axis=0)
    else:
        return {"error": "No valid words in the input text"}
    
    # Pilih model berdasarkan input dari user (KNN atau SVM)
    if chosen_model == "knn":
        prediction = knn.predict([text_vector])
    elif chosen_model == "svm":
        prediction = svm.predict([text_vector])
    else:
        raise HTTPException(status_code=400, detail="Model yang dipilih tidak valid. Pilih antara 'knn' atau 'svm'.")

    # Render halaman dengan hasil prediksi
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "text": text,
            "predicted_sentiment": prediction[0]
        }
    })

# Untuk menjalankan aplikasi FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
