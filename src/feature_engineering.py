import nltk

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
from rich.console import Console
from rich.progress import track
from rich import print as rprint

console = Console()

def create_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates sentiment features for title and description columns
    """
    # Initialize VADER sentiment analyzer
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # Combine title and subtitle
    df['combined_title'] = df.apply(
        lambda x: x['title'] + ' ' + x['subtitle'] 
        if pd.notna(x['subtitle']) else x['title'], 
        axis=1
    )
    
    # Calculate sentiment scores for combined title
    title_scores = df['combined_title'].apply(sia.polarity_scores)
    df['title_positive'] = title_scores.apply(lambda x: x['pos'])
    df['title_negative'] = title_scores.apply(lambda x: x['neg'])
    df['title_neutral'] = title_scores.apply(lambda x: x['neu'])
    df['title_compound'] = title_scores.apply(lambda x: x['compound'])
    
    # Calculate sentiment scores for description
    description_scores = df['description'].apply(
        lambda x: sia.polarity_scores(x) if isinstance(x, str) 
        else {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}
    )
    df['description_positive'] = description_scores.apply(lambda x: x['pos'])
    df['description_negative'] = description_scores.apply(lambda x: x['neg'])
    df['description_neutral'] = description_scores.apply(lambda x: x['neu'])
    df['description_compound'] = description_scores.apply(lambda x: x['compound'])
    
    return df

def create_authors_embeddings(df: pd.DataFrame, authors_col: str, vector_size: int = 50) -> pd.DataFrame:
    """
    Crea un único vector de embeddings para autores por fila
    """
    authors_list = df[authors_col].astype(str).apply(lambda x: x.split(";"))
    model = Word2Vec(sentences=authors_list, vector_size=vector_size, min_count=1, workers=4)
    
    def get_author_vector(author_names):
        vectors = []
        for name in author_names:
            name = name.strip()
            if name in model.wv:
                vectors.append(model.wv[name])
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
    
    df['author_vector'] = authors_list.apply(get_author_vector)
    return df

def create_category_vector(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """
    Crea un vector único para categorías por fila
    """
    categories = df[category_col].str.get_dummies(sep=';')
    df['category_vector'] = categories.values.tolist()
    return df

def feature_engineering(input_path: str, output_path: str):
    """
    Proceso de feature engineering con visualización del progreso
    """
    console.rule("[bold purple]Feature Engineering Process")
    
    df = pd.read_csv(input_path)
    rprint(f"[bold cyan]📊 Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    rprint("[bold blue]⚡ Tranformando algunos datos...")
    
    df['num_pages_log'] = np.log10(df['num_pages'])
    df['year_group'] = (df['published_year'] // 5) * 5
    
    rprint("[bold blue]⚡ Generando embeddings de autores...")
    df = create_authors_embeddings(df, 'authors')
    
    rprint("[bold blue]🎯 Vectorizando categorías...")
    df = create_category_vector(df, 'categories')
    
    rprint("[bold blue]📊 Analizando sentimientos del título y descripción...")
    df = create_sentiment_features(df)
    
    df.to_csv(output_path, index=False)
    rprint("[bold green]🚀 Feature engineering completado!")
    
    console.rule("[bold purple]Proceso Finalizado")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input", required=True, help="Ruta de entrada")
    parser.add_argument("--output", required=True, help="Ruta de salida")
    args = parser.parse_args()
    
    feature_engineering(args.input, args.output)
