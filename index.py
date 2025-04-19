from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import traceback
from fastapi.middleware.cors import CORSMiddleware
import psutil
import os

nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

model_path = "./model/model.keras"
tokenizer_path = "./model/tokenizer.pkl"

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model(model_path)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

origins = [
    "http://localhost:3000",  
    "https://fakenewsflush.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


class Article(BaseModel):
    title: str
    content: str


print_memory_usage()

def preprocess_data(title: str, content: str):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(string.punctuation)

    url_pattern = re.compile(r'http\S+')
    punctuation_trans = str.maketrans('', '', string.punctuation + '“”’')
    digits_trans = str.maketrans('', '', string.digits)

    # Concatenate relevant data
    text = f"{title} {content}"

    # Convert to lowercase and remove URLs
    text = url_pattern.sub('', text.lower())

    # Remove punctuation and digits
    text = text.translate(punctuation_trans)
    text = text.translate(digits_trans)

    # Remove stopwords and lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

    # Create sequences and pad according to max_len established during training
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=488, padding='post')

    return padded

@app.head("/")
@app.get("/")
def read_root():
    return {"status": "OK"}

@app.post("/predict")
async def predict_article(article: Article):
    try:
        # Preprocess data
        clean_data = preprocess_data(article.title, article.content)

        print_memory_usage()

        # Perform inference
        prediction = model.predict(clean_data)
        label = "REAL" if prediction[0][0] > 0.5 else "FAKE"
        print_memory_usage()
        return {"title": article.title, "prediction": label}
    except Exception as e:
        error_trace = traceback.format_exc()
        print("Error occurred during prediction:\n", error_trace)
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")