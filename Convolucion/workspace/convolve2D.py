from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
ruta_recursos='/home/diego/GitHub/Analisis_Se-al_2025_1/Convolucion/workspace/Recursos/'
# Ruta de la imagen
image_path = '/home/diego/GitHub/Analisis_Se-al_2025_1/Convolucion/workspace/Recursos/I.jpeg'
# Abre la imagen con Pillow
try:
    image = Image.open(image_path)
    image_array = np.array(image)  # Convierte la imagen a un array de numpy si es necesario
    print("Imagen cargada exitosamente.")
except Exception as e:
    print(f"Error al abrir la imagen: {e}")
print(type(image))
image_resc=np.array(image.resize((512,512)))

# Imprime la dimensión del arreglo
print(f"Dimensiones del arreglo de la imagen: {image_array.shape}")

red_image=image_array[:,:,0]
green_image=image_array[:,:,1]
blue_image=image_array[:,:,2]
red_image_resized=image_resc[:,:,0]
print(red_image.shape)
print(red_image[4,4])
print(red_image.dtype)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(red_image,cmap='Reds')
plt.title("Image en 2D (canal rojo)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(green_image,cmap='Greens')
plt.title("Image en 2D (canal verde)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(blue_image,cmap='Blues')
plt.title("Image en 2D (canal azul)")
plt.axis('off')

plt.savefig(f'{ruta_recursos}color_layers.png', dpi=300, bbox_inches='tight')




plt.figure(figsize=(12,5))
plt.imshow(red_image_resized,cmap='gray')
plt.title("Image en 2D (canal rojo rescalado)")
plt.axis('off')
plt.savefig(f'{ruta_recursos}gray_resized_image.png', dpi=300, bbox_inches='tight')




#creamos un filtro promedio
filter_mean5=np.ones((5,5))/5

#convolucionamos las dos imagenes
image_filtered=convolve2d(red_image_resized,filter_mean5,mode='same')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(red_image_resized,cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_filtered,cmap='gray')
plt.title("Imagen filtrada (convolucion2D)")
plt.axis('off')
plt.savefig(f'{ruta_recursos}gray_image_filtered.png', dpi=300, bbox_inches='tight')


# Creamos un filtro gaussiano 5x5
gaussian_filter_5x5 = np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]]) / 273

# Aplicamos la convolución con el filtro gaussiano
image_gaussian_filtered = convolve2d(red_image_resized, gaussian_filter_5x5, mode='same')

# Mostramos la imagen filtrada con el filtro gaussiano
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(red_image_resized, cmap='gray')
plt.title("Imagen original (canal rojo rescalado)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_gaussian_filtered, cmap='gray')
plt.title("Imagen filtrada (filtro gaussiano 5x5)")
plt.axis('off')
plt.savefig(f'{ruta_recursos}gray_image_filteredG.png', dpi=300, bbox_inches='tight')



# Creamos un filtro gaussiano 3x3
gaussian_filter_3x3 = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16

# Aplicamos la convolución con el filtro gaussiano 3x3
image_gaussian_filtered_3x3 = convolve2d(red_image_resized, gaussian_filter_3x3, mode='same')

# Mostramos la imagen filtrada con el filtro gaussiano 3x3
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(red_image_resized, cmap='gray')
plt.title("Imagen original (canal rojo rescalado)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_gaussian_filtered_3x3, cmap='gray')
plt.title("Imagen filtrada (filtro gaussiano 3x3)")
plt.axis('off')
plt.savefig(f'{ruta_recursos}gray_image_filteredG1.png', dpi=300, bbox_inches='tight')



# Filtros Sobel para detección de bordes
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])

# Aplicamos la convolución con el filtro Sobel en X
image_sobel_x = convolve2d(red_image_resized, sobel_x, mode='same')

# Aplicamos la convolución con el filtro Sobel en Y
image_sobel_y = convolve2d(red_image_resized, sobel_y, mode='same')

# Magnitud del gradiente
image_sobel_magnitude = np.sqrt(image_sobel_x**2 + image_sobel_y**2)

# Mostramos los resultados
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_sobel_x, cmap='gray')
plt.title("Filtro Sobel (X)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_sobel_y, cmap='gray')
plt.title("Filtro Sobel (Y)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_sobel_magnitude, cmap='gray')
plt.title("Magnitud del gradiente (Sobel)")
plt.axis('off')

plt.savefig(f'{ruta_recursos}gray_image_filteredS.png', dpi=300, bbox_inches='tight')



laplacian_filter_1=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
laplacian_filter_2=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])


image_laplacian_1=convolve2d(red_image_resized,laplacian_filter_1,mode='same')
image_laplacian_2=convolve2d(red_image_resized,laplacian_filter_2,mode='same')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(image_laplacian_1,cmap='gray')
plt.title("Filtro laplaciano 3x3")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_laplacian_2,cmap='gray')
plt.title("Filtro laplaciano 3x3")
plt.axis('off')

plt.savefig(f'{ruta_recursos}gray_image_filteredL1.png', dpi=300, bbox_inches='tight')



#realce

sharpen_filter_1=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen_filter_2=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

lambda1=10
lambda2=10

image_sharpen_1=lambda1*convolve2d(red_image_resized,sharpen_filter_1,mode='same')+red_image_resized
image_sharpen_2=lambda2*convolve2d(red_image_resized,sharpen_filter_2,mode='same')+red_image_resized

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image_sharpen_1,cmap='gray')
plt.title("filtro de realce1")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_sharpen_2,cmap='gray')
plt.title("filtro de realce2")
plt.axis('off')

plt.savefig(f'{ruta_recursos}gray_image_filteredS1.png', dpi=300, bbox_inches='tight')








