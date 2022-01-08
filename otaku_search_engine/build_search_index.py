
try:
    from sentence_transformers import SentenceTransformer
    import scipy
    import os
    import pandas as pd
    import pickle
    import argparse
except ImportError:
    print("Please install the required packages: sentence_transformers, scipy, pandas, and pickle")

# Setting up argument parser
parser = argparse.ArgumentParser(description='Build search index')

# Directory arguments with defaults
parser.add_argument('--data_file', type=str, default='Anime_Top10000.csv', help='Data file')
parser.add_argument('--data_file_cleaned', type=str, default='Anime_Top10000_cleaned.csv', help='Text data directory')
parser.add_argument('--text_data_dir', type=str, default='data', help='Text data directory')
parser.add_argument('--embedding_file', type=str, default='sentence_embeddings.pkl', help='Embedding file')
parser.add_argument('--embedding_data_dir', type=str, default='embeddings', help='Text data directory')


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


# DATA PROVENANCE
# CHECK SOURCE DATA AND CLEAN
# Data file and previous ata provenance available AT https://www.kaggle.com/thomaskonstantin/top-10000-anime-movies-ovas-and-tvshows
def process_data(data_file, text_data_dir = 'data'):
    TEXT_DATA_DIR = text_data_dir # "data"
    DATA_FILE = data_file # "Anime_Top10000.csv"
    raw_df = read_csv(os.path.join(TEXT_DATA_DIR, DATA_FILE))
    raw_df_clean = raw_df.drop_duplicates(subset=['Synopsis'])  # Drops duplicate synopsis
    mask = (raw_df['Synopsis'].str.len() > 180) # Filtering out shows with short synopsis
    raw_df_clean = raw_df[mask] # Longer texts to train on
    #raw_df_clean = raw_df_clean.dropna(subset=['Synopsis']) # Drops duplicate synopsis, NO NULL Values in original dataset
    #raw_df_clean = raw_df_clean.drop_duplicates(subset=['Anime_Name']) # Drops duplicate show, DUPLICATE Synopsis solves this issue
    print(raw_df_clean.head(5))
    print(len(raw_df_clean.index))
    raw_df_clean.to_csv(DATA_FILE.split('.')[0] + '_cleaned.csv')


# A corpus is a list with documents split by sentences.
def train_embeddings(data_file, text_data_dir = 'data'):
    TEXT_DATA_DIR = text_data_dir  # "data"
    DATA_FILE = data_file  # "Anime_Top10000_cleaned.csv"
    
    # Load the BERT model.
    # More models available under under Natural Language Inference (NLI) https://github.com/UKPLab/sentence-transformers/docs/pretrained-models/nli-models.md
    
    # Loading BERT pretrained model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Check max sequence length by number of words
    print("Max Sequence Length:", model.max_seq_length)
    
    input_df = read_csv(os.path.join(TEXT_DATA_DIR, DATA_FILE))
    print("Processing... " + str(len(input_df.index)) + " rows.")

    sentences = input_df['Synopsis'].values.tolist()

    print(sentences[0])
    #sentences = ['synopsis 1',
    #             'synopsis 2',
    #             'synopsis 3',
    #             ...
    #             'synopsis x']

    # Each sentence is encoded as a 1-D vector with 78 columns
    sentence_embeddings = model.encode(sentences)

    print('Embeddings genereated...')
    print('Single BERT embedding vector - length', len(sentence_embeddings[0]))

    #print('Single BERT embedding vector (row 1)', sentence_embeddings[0])
    return sentences, sentence_embeddings

def save_embeddings(sentences, embeddings, embedding_file = 'sentence_embeddings.pkl', embedding_data_dir = 'embeddings'):
    EMBEDDING_FILE = embedding_file # "sentence_embeddings.pkl"
    EMBEDDING_DATA_DIR = embedding_data_dir # "embeddings"
    # Save embeddings
    with open(os.path.join(EMBEDDING_DATA_DIR, EMBEDDING_FILE), 'wb') as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    if os.path.exists(embedding_file):
        print("Embedding file saved successfully")
        return None
    
    print("Embedding file failed to save")
    return None

def main(args):
    """
    Main function for training, evaluating and checkpointing.

    Args:
        args (argparse.Namespace): arguments that can be provided by the user
    """
    # Check for directories
    if not os.path.exists(args.file_data_dir):
        os.makedirs(args.file_data_dir)
    if not os.path.exists(args.embedding_data_dir):
        os.makedirs(args.embedding_data_dir)
    
    # Check for embedding file
    if os.path.exists(os.path.join(args.embedding_data_dir, args.embedding_file)):
        print("Embedding file already exists, skipping training")

    # Check for cleaned data file
    if os.path.exists(os.path.join(args.text_data_dir, args.data_file_cleaned)):
        print("Data file already exists, skipping training")
        return None

    # Process data
    print("Cleaned data file not found. Processing data...")
    if not os.path.exists(os.path.join(args.file_data_dir, args.data_file)):
        print("Data file not found. Please manually download at https://www.kaggle.com/thomaskonstantin/top-10000-anime-movies-ovas-and-tvshows")
        print("Exiting...")
        return None
    process_data(args.data_file, args.text_data_dir)
    print("Data file processed.")
    os.makedirs(args.text_data_dir)

    if not os.path.exists(os.path.join(args.embedding_data_dir, args.embedding_file)):
        print("Embedding file not found. Processing data...")
        sentences, embeddings = train_embeddings(args.data_file_cleaned, args.text_data_dir)
        save_embeddings(sentences, embeddings, args.embedding_file, args.embedding_data_dir)
        print("Embedding file processed.")
    
    print("Semantic Search Index Built. Done.")

if __name__ == '__main__':
    main(parser.parse_args())
