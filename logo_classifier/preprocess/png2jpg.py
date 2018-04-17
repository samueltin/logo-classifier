import os
from PIL import Image

IMAGE_ROOT_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'images', 'val')
IMAGE_SUB_DIR='IBM'

def _convert(img_dir):
    image_list = os.listdir(img_dir)
    for image_file in image_list:
        if image_file != '.DS_Store' and image_file.endswith('.png'):
            full_file = os.path.join(img_dir, image_file)
            new_image_file = image_file.split('.')[0]+'.jpg'
            full_new_file = os.path.join(img_dir, new_image_file)
            im = Image.open(full_file)
            rgb_im = im.convert('RGB')
            rgb_im.save(full_new_file)

def main():
    _convert(os.path.join(IMAGE_ROOT_DIR,'Apple'))
    _convert(os.path.join(IMAGE_ROOT_DIR, 'IBM'))

if __name__ == "__main__":
    main()