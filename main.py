# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import integral_image
from PIL import Image


def main():
    # Usando PIL abrimos la imagen y la guardamos en una variable
    img = Image.open('source/image1.png')
    # Para trabjar con la imagen convertimos la imagen en un arreglo
    # para manipular los pixeles
    img_array = np.array(img)
    # Usando la funcion integral_image de skicit-image
    # sumamos todos los cuadrantes de la imagen
    my_intergral_image = integral_image(img_array)
    # Para poder mostrar la imagen debemos realizar
    # la conversión del arreglo a una imagen
    # transformando los datos del arreglo a pixeles nuevamente
    # usando el tipo de dato uint8 de numpy
    image_real_integral = Image.fromarray(
        np.uint8(my_intergral_image)
    )
    # Mostramos la primera imagen
    img.show()
    # Finalmente mostramos la Imagen Integral generada
    image_real_integral.show()
    # image_real_integral.save('image_integral_1.png')


if __name__ == '__main__':
    main()
