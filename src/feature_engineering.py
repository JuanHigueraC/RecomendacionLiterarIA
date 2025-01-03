import pandas as pd
import numpy as np
from rich.console import Console
from gensim.models import Word2Vec

console = Console()

def apply_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aplica una transformación logarítmica (base 10) a una columna numérica.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna numérica a transformar.

    Returns:
        pd.DataFrame: El DataFrame original con una nueva columna `<column>_log`.
    """
    console.print(f"[yellow]Aplicando transformación log a {column}...[/yellow]")
    df[f'{column}_log'] = np.log10(df[column])
    return df

def create_year_groups(df: pd.DataFrame, year_column: str, interval: int = 5) -> pd.DataFrame:
    """
    Crea grupos de años a partir de una columna de año con un intervalo especificado.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        year_column (str): El nombre de la columna de año en el DataFrame.
        interval (int, optional): El intervalo de agrupación. Por defecto es 5.

    Returns:
        pd.DataFrame: El DataFrame original con una nueva columna `year_group`.
    """
    console.print(f"[yellow]Creando grupos de {interval} años desde la columna {year_column}...[/yellow]")
    df['year_group'] = (df[year_column] // interval) * interval
    return df

def create_one_hot_encoding(df: pd.DataFrame, column: str, prefix: str = None) -> pd.DataFrame:
    """
    Genera variables dummies (one-hot encoding) para una columna categórica.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna categórica.
        prefix (str, optional): Prefijo para las columnas dummies. Si no se especifica,
            se usa el nombre de la columna.

    Returns:
        pd.DataFrame: El DataFrame original con las columnas dummies concatenadas.
    """
    console.print(f"[yellow]Generando one-hot encoding para {column}...[/yellow]")
    prefix = prefix or column
    dummies = pd.get_dummies(df[column], prefix=prefix)
    return pd.concat([df, dummies], axis=1)

def create_authors_embeddings(
    df: pd.DataFrame,
    authors_col: str,
    vector_size: int = 50,
    min_count: int = 1,
    workers: int = 4
) -> pd.DataFrame:
    """
    Crea embeddings de Word2Vec para la columna de autores y los concatena como nuevas columnas.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        authors_col (str): El nombre de la columna que contiene los autores.
                           Se asume que los autores están separados por ';'.
        vector_size (int, optional): Dimensión del embedding. Por defecto es 50.
        min_count (int, optional): Frecuencia mínima para entrenar Word2Vec. Por defecto es 1.
        workers (int, optional): Número de threads para entrenar Word2Vec. Por defecto es 4.

    Returns:
        pd.DataFrame: El DataFrame original con las nuevas columnas de embedding para cada autor.
    """
    console.print(f"[yellow]Creando embeddings de Word2Vec para la columna '{authors_col}'...[/yellow]")

    # Separamos los autores y los convertimos a listas de strings
    authors_list = df[authors_col].astype(str).apply(lambda x: x.split(";"))

    # Entrenamos el modelo de Word2Vec
    console.print(f"[cyan]Entrenando modelo Word2Vec con vector_size={vector_size}, min_count={min_count}...[/cyan]")
    model = Word2Vec(
        sentences=authors_list,
        vector_size=vector_size,
        min_count=min_count,
        workers=workers
    )

    # Definimos una función helper para obtener el embedding promedio de los autores
    def embed_authors(author_names):
        """
        Obtiene la representación vectorial promedio para una lista de autores.

        Args:
            author_names (list): Lista de autores en formato string.

        Returns:
            np.ndarray: Un vector de dimensión `vector_size` con el promedio de los embeddings.
        """
        vectors = []
        for name in author_names:
            name = name.strip()
            if name in model.wv:
                vectors.append(model.wv[name])
            else:
                # Si no está en el vocabulario, devolvemos vector de ceros o lo ignoramos
                vectors.append(np.zeros(vector_size))
        if len(vectors) == 0:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    # Creamos una lista con los embeddings promedio para cada fila
    console.print(f"[cyan]Calculando embeddings promedio para cada fila...[/cyan]")
    embeddings_list = authors_list.apply(embed_authors)

    # Convertimos la lista de arrays en un DataFrame con una columna por dimensión
    embedding_columns = [f"{authors_col}_emb_{i}" for i in range(vector_size)]
    embedding_df = pd.DataFrame(embeddings_list.to_list(), columns=embedding_columns, index=df.index)

    # Concatena las nuevas columnas de embeddings al DataFrame original
    df = pd.concat([df, embedding_df], axis=1)
    console.print(f"[green]Embeddings para la columna '{authors_col}' creados con éxito.[/green]")

    return df

def feature_engineering(input_path: str, output_path: str):
    """
    Ejecuta el proceso de feature engineering, que incluye:
    - Carga de datos
    - Transformaciones numéricas (transformación log)
    - Creación de grupos de año
    - One-hot encoding para categorías
    - Creación de embeddings Word2Vec para la columna de autores

    Args:
        input_path (str): Ruta del archivo CSV de entrada.
        output_path (str): Ruta donde se guardará el archivo CSV resultante.

    Returns:
        None
    """
    console.print("[bold green]Iniciando proceso de feature engineering[/bold green]")
    
    console.print("[yellow]Cargando dataset...[/yellow]")
    df = pd.read_csv(input_path)
    console.print(f"Se cargaron {len(df)} filas de datos.")

    # Aplicar transformaciones
    df = apply_log_transform(df, 'num_pages')
    df = create_year_groups(df, 'published_year')
    df = create_one_hot_encoding(df, 'categories')

    # Crear embeddings para la columna 'authors'
    df = create_authors_embeddings(df, 'authors', vector_size=50, min_count=1, workers=4)
    
    console.print(f"[yellow]Guardando el dataset transformado en {output_path}...[/yellow]")
    df.to_csv(output_path, index=False)
    
    console.print("[bold green]¡Feature engineering completado exitosamente! ✨[/bold green]")
    console.print(f"[blue]Tamaño final del dataset: {df.shape}[/blue]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input", required=True, help="Ruta de los datos procesados")
    parser.add_argument("--output", required=True, help="Ruta del archivo con features generados")
    args = parser.parse_args()

    feature_engineering(args.input, args.output)
