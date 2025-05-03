import pandas as pd
RESOURCE_PATH = "/home/diego/GitHub/Analisis_Se-al_2025_1/Quiz_1/workspace/recursos/mushroom/agaricus-lepiota.data"

# Load the dataset into a pandas DataFrame
def load_dataset(resource_path):
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    df = pd.read_csv(resource_path, header=None, names=column_names)
    return df

# Load the dataset and print a sample
if __name__ == "__main__":
    
    # a) Numero de Caracteristicas del dataset
    dataset = load_dataset(RESOURCE_PATH)
    num_features=len(dataset.columns)-1
    print(f"Number of Features is:{num_features}")

    # b) Nombre iniciales tipos, renombramiento
    print("Nombres iniciales de las columnas:",list(dataset.columns))
    print("Tipos de las columnas:")
    print(dataset.dtypes)

    #renombrar las columnas
    new_columns=list(dataset.columns)
    new_columns[0]="edibility"

    dataset.columns=new_columns

