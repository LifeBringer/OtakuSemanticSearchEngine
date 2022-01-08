import os
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import scipy
import pickle
import pandas as pd
import keras.backend as tb
from keras.models import model_from_json


# Load the BERT model. 
model = SentenceTransformer('bert-base-nli-mean-tokens')

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)

@app.route("/")
def hello():
	return TEMPLATE_DIR

# A corpus is a list with documents split by sentences.
TEXT_DATA_DIR = 'data'
CLEANED_DATA_FILE_NAME = "Anime_Top10000_cleaned.csv"


# Setup read csv funcition
def read_csv(filepath):
    if os.path.splitext(filepath)[1] != '.csv':
        return None
    seps = [',', ';', '\t']  # ',' is default
    encodings = [None, 'utf-8', 'ISO-8859-1', 'utf-16', 'ascii']  # None is default
    for sep in seps:
        for encoding in encodings:
            try:
                return pd.read_csv(filepath, encoding=encoding, sep=sep)
            except Exception:
                pass
    raise ValueError("{!r} is has no encoding in {} or seperator in {}".format(filepath, encodings, seps))

CLEANED_DATA_FILE_PATH = os.path.join(TEXT_DATA_DIR, CLEANED_DATA_FILE_NAME)
input_df = read_csv(CLEANED_DATA_FILE_PATH)
print(input_df.head(20))

# Load embeddings
def load_embeddings(embedding_file_name = 'otaku_embeddings.pkl', embedding_file_dir = 'embeddings'):
    EMBEDDING_FILE_PATH = os.path.join(embedding_file_dir, embedding_file_name)

    # Load embeddings
    with open(EMBEDDING_FILE_PATH, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']

    return stored_sentences, stored_embeddings

stored_sentences, stored_embeddings = load_embeddings()

def performSearch(query):
    queries = [query]
    query_embeddings = model.encode(queries)

    # Find top 5 closest matches for query
    num_top_matches = 5 #@param {type: 'number', min: 1, max: 5, step: 1}

    print("Searching for:", query)
    results = []
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], stored_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
    
    return results[:num_top_matches]


@app.route("/index.html",	 methods=['GET', 'POST'])
def index():
    errors = []
    query = ''
    if(request.method == "POST"):
        query = request.form.get('query')
        results = performSearch(query)
        return render_template('index.html', query=query, results=results, sentences=input_df)
    else:
        return render_template('index.html', errors=errors, review="", results=None)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True, threaded=False)
