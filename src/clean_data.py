import pandas as pd
import dvc.api
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box
from rich import print as rprint

console = Console()

def remove_missing_values(df):
    columns_to_check = ['isbn13', 'title', 'authors', 'categories', 'description', 'published_year', 'num_pages']
    return df.dropna(subset=columns_to_check)

def remove_zero_num_pages(df):
    mask = df["num_pages"]==0
    return df[~mask]

def create_quality_table(df):
    with console.status("[bold blue]📊 Calculando métricas de calidad...") as status:
        quality_table = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Count': df.count(),
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df)) * 100,
            'Unique Values': df.nunique(),
            'Unique %': (df.nunique() / len(df)) * 100
        })
        
        quality_table['Missing %'] = quality_table['Missing %'].round(2)
        quality_table['Unique %'] = quality_table['Unique %'].round(2)
        
        rprint("[bold green]✨ Métricas calculadas exitosamente!")
    return quality_table

def display_quality_comparison(before_df, after_df):
    console.rule("[bold purple]Comparación de Calidad de Datos")
    
    before_quality = create_quality_table(before_df)
    after_quality = create_quality_table(after_df)
    
    table = Table(
        title="Data Quality Comparison",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        title_style="bold blue"
    )
    
    table.add_column("Column Name", style="cyan")
    table.add_column("Before Missing %", style="red")
    table.add_column("After Missing %", style="green")
    table.add_column("Before Unique %", style="yellow")
    table.add_column("After Unique %", style="blue")
    
    for col in track(before_df.columns, description="[bold blue]Generando tabla comparativa"):
        before_missing = f"{before_quality.loc[before_quality['Column'] == col, 'Missing %'].iloc[0]}%"
        after_missing = f"{after_quality.loc[after_quality['Column'] == col, 'Missing %'].iloc[0]}%"
        before_unique = f"{before_quality.loc[before_quality['Column'] == col, 'Unique %'].iloc[0]}%"
        after_unique = f"{after_quality.loc[after_quality['Column'] == col, 'Unique %'].iloc[0]}%"
        
        table.add_row(col, before_missing, after_missing, before_unique, after_unique)
    
    console.print(table)

def clean_data(input_path, output_path):
    console.rule("[bold purple]Proceso de Limpieza de Datos")
    
    with console.status("[bold blue]📚 Cargando dataset...") as status:
        df = pd.read_csv(input_path, encoding='utf-8')
        rprint(f"[bold cyan]📊 Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    before_df = df.copy()
    
    with console.status("[bold blue]🧹 Limpiando datos...") as status:
        df = remove_missing_values(df)
        df = remove_zero_num_pages(df)
        rprint(f"[bold green]✨ Limpieza completada! Nuevas dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    display_quality_comparison(before_df, df)
    
    with console.status("[bold blue]💾 Guardando resultados...") as status:
        df.to_csv(output_path, index=False)
        rprint("[bold green]🚀 Datos limpios guardados exitosamente!")
    
    console.rule("[bold purple]Proceso Finalizado")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Limpieza de datos")
    parser.add_argument("--input", required=True, help="Ruta de los datos originales (DVC-tracked)")
    parser.add_argument("--output", required=True, help="Ruta de los datos procesados")
    args = parser.parse_args()

    clean_data(args.input, args.output)
