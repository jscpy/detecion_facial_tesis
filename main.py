# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import integral_image
from PIL import Image


def main():
    img = Image.open('source/image1.png')
    img_array = np.array(img)
    my_intergral_image = integral_image(img_array)
    image_real_integral = Image.fromarray(
        np.uint8(my_intergral_image)
    )
    # Imagen real
    img.show()
    # Imagen Integral
    image_real_integral.show()
    # image_real_integral.save('image_integral_1.png')


if __name__ == '__main__':
    main()
