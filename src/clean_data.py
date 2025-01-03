import pandas as pd
import dvc.api
from rich.console import Console
from rich.table import Table
from rich import box

def remove_missing_values(df):
    columns_to_check = ['isbn13', 'title', 'authors', 'categories', 'description', 'published_year', 'num_pages']
    return df.dropna(subset=columns_to_check)

def create_quality_table(df):
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
    
    return quality_table

def display_quality_comparison(before_df, after_df):
    console = Console()
    
    before_quality = create_quality_table(before_df)
    after_quality = create_quality_table(after_df)
    
    table = Table(
        title="Data Quality Comparison",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        title_style="bold blue"
    )
    
    # Add columns
    table.add_column("Column Name", style="cyan")
    table.add_column("Before Missing %", style="red")
    table.add_column("After Missing %", style="green")
    table.add_column("Before Unique %", style="yellow")
    table.add_column("After Unique %", style="blue")
    
    # Add rows
    for col in before_df.columns:
        before_missing = f"{before_quality.loc[before_quality['Column'] == col, 'Missing %'].iloc[0]}%"
        after_missing = f"{after_quality.loc[after_quality['Column'] == col, 'Missing %'].iloc[0]}%"
        before_unique = f"{before_quality.loc[before_quality['Column'] == col, 'Unique %'].iloc[0]}%"
        after_unique = f"{after_quality.loc[after_quality['Column'] == col, 'Unique %'].iloc[0]}%"
        
        table.add_row(col, before_missing, after_missing, before_unique, after_unique)
    
    console.print(table)

def clean_data(input_path, output_path):
    #with dvc.api.open(input_path, mode='r') as f:
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Store original data quality
    before_df = df.copy()
    
    # Clean data
    df = remove_missing_values(df)
    
    # Display quality comparison
    display_quality_comparison(before_df, df)
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Limpieza de datos")
    parser.add_argument("--input", required=True, help="Ruta de los datos originales (DVC-tracked)")
    parser.add_argument("--output", required=True, help="Ruta de los datos procesados")
    args = parser.parse_args()

    clean_data(args.input, args.output)
