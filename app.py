import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import gensim
import nltk
import compress_pickle
import os

# Inisialisasi FastAPI
app = FastAPI()

# Setup Jinja2 untuk template HTML
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Download NLTK stopwords dan tokenizer
nltk.download('punkt')
nltk.download('stopwords')

# Memuat model FastText dari file .pkl.gz yang terkompresi
model_vec = compress_pickle.load('models/cc.id.300.pkl.gz')

# Muat model klasifikasi
knn = pickle.load(open('models/KNN+GridSearch.pkl', 'rb'))
svm = pickle.load(open('models/SVM+GridSearch.pkl', 'rb'))

# Fungsi preprocessing teks
def preprocess_text(text):
    # Hapus tanda baca dan ubah menjadi huruf kecil
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)

    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]

    # Stemming menggunakan Sastrawi
    stemmer = StemmerFactory().create_stemmer()
    words = [stemmer.stem(word) for word in words]

    return words

# Endpoint untuk menampilkan halaman utama (form input)
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint untuk menerima form input dan memberikan prediksi sentimen
@app.post("/predict")
async def predict_sentiment(request: Request, text: str = Form(...), chosen_model: str = Form(...)):
    clean_words = preprocess_text(text)

    # Ambil vektor untuk setiap kata yang ada dalam model FastText
    word_vectors = [model_vec[word] for word in clean_words if word in model_vec]
    
    if word_vectors:
        text_vector = np.mean(word_vectors, axis=0)
    else:
        return {"error": "No valid words in the input text"}
    
    # Pilih model berdasarkan input dari user (KNN atau SVM)
    if chosen_model.lower() == "knn":
        prediction = knn.predict([text_vector])
    elif chosen_model.lower() == "svm":
        prediction = svm.predict([text_vector])
    else:
        raise HTTPException(status_code=400, detail="Model yang dipilih tidak valid. Pilih antara 'knn' atau 'svm'.")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "text": text,
            "predicted_sentiment": "Positive" if prediction[0] == 1 else "Negative",
            "model": chosen_model
        }
    })

# Untuk menjalankan aplikasi FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
