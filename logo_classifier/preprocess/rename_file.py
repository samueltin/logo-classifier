import os

IMAGE_ROOT_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'images')
IMAGE_SUB_DIR='Apple'
START_INDEX = 1


def _rename_file(sub_dir, start_idx):
    full_dir = os.path.join(IMAGE_ROOT_DIR, sub_dir)
    print(full_dir)
    image_list = os.listdir(full_dir)

    for image_file in image_list:
        print (image_file)
        full_image_file = os.path.join(full_dir, image_file)
        print(full_image_file)
        _, file_extension = os.path.splitext(full_image_file)
        if image_file != '.DS_Store':
            os.rename(full_image_file, os.path.join(full_dir, sub_dir+'_'+str(start_idx).zfill(8)+file_extension))
            start_idx+=1


def main():
    _rename_file(IMAGE_SUB_DIR, START_INDEX)

if __name__ == "__main__":
    main()