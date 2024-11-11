import pandas as pd 
import re 

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_lyrics(lyrics):
    cleaned = re.sub(r"\[.*?\]", "", lyrics) # Remove sections like [Intro], [Verse]
    return re.sub(r"[^a-zA-Z\s]", "", cleaned).strip().lower() # Letters only

def preprocess_data(file_path):
    df = load_data(file_path)
    df['cleaned_lyrics'] = df['lyrics_text'].dropna().apply(clean_lyrics)
    return df[['track_name', 'cleaned_lyrics']]


