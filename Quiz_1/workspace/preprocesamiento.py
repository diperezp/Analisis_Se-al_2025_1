import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gensim.downloader as api
from gensim.models import KeyedVectors

# ------------------ Método 1: PCA (Análisis de Componentes Principales) ------------------
# Usaremos el conjunto de datos MNIST (dígitos) para ilustrar PCA
def illustrate_pca():
    # Cargar el conjunto de datos de dígitos
    digits = load_digits()
    X = digits.data  # Características: imágenes de 8x8 píxeles (64 dimensiones)
    y = digits.target  # Etiquetas (0-9)

    # Estandarizar los datos (media=0, desviación estándar=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA para reducir de 64 dimensiones a 2 dimensiones (para visualización)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Visualizar los datos transformados
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Visualización de MNIST con PCA (reducción a 2D)')
    plt.grid(True)
    plt.show()

    # Imprimir cuánta varianza explican los componentes principales
    print(f"Varianza explicada por los 2 componentes principales: {sum(pca.explained_variance_ratio_):.2f}")

# ------------------ Método 2: Embeddings (usando Word2Vec preentrenado) ------------------
# Usaremos embeddings preentrenados de Word2Vec para palabras
def illustrate_embeddings():
    # Cargar un modelo preentrenado de Word2Vec (esto puede tomar un momento la primera vez)
    print("Cargando modelo Word2Vec preentrenado...")
    model = api.load("word2vec-google-news-300")  # Modelo de 300 dimensiones

    # Seleccionar algunas palabras para visualizar
    words = ['dog', 'cat', 'house', 'building', 'car', 'truck']
    word_vectors = np.array([model[word] for word in words])

    # Reducir los embeddings de 300 dimensiones a 2D para visualización usando PCA
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)

    # Visualizar las palabras en el espacio 2D
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1], marker='o')
        plt.text(word_vectors_2d[i, 0] + 0.3, word_vectors_2d[i, 1], word, fontsize=12)
    
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.title('Visualización de Embeddings de Palabras (Word2Vec)')
    plt.grid(True)
    plt.show()

    # Mostrar la similitud entre algunas palabras
    print(f"Similitud entre 'dog' y 'cat': {model.similarity('dog', 'cat'):.3f}")
    print(f"Similitud entre 'house' y 'building': {model.similarity('house', 'building'):.3f}")
    print(f"Similitud entre 'dog' y 'house': {model.similarity('dog', 'house'):.3f}")

# ------------------ Ejecutar ambos métodos ------------------
if __name__ == "__main__":
    print("=== Ilustración de PCA ===")
    illustrate_pca()
    
    print("\n=== Ilustración de Embeddings ===")
    illustrate_embeddings()