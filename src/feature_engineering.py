import nltk

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
from rich.console import Console
from rich.progress import track
from rich import print as rprint

console = Console()

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
    
    rprint("[bold blue]🗺️ Vectorizando titulos y descripciones...")
    df = create_category_vector(df, 'categories')
    
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
