import cv2
import os

IMAGE_ROOT_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'images')
IMAGE_SUB_DIR=['train', 'val', 'test']
IMAGE_SUB_DIR=['temp']


def _resize_image(sub_dirs, img_w, img_h):
    for sub_dir in sub_dirs:
        path1 = os.path.join(IMAGE_ROOT_DIR, sub_dir)
        dir_list = os.listdir(path1)
        for dir in dir_list:
            if dir != '.DS_Store':
                path2 = os.path.join(path1, dir)
                file_list = os.listdir(path2)
                for file in file_list:
                    if file != '.DS_Store':
                        filename = os.path.join(path2, file)
                        oriimage = cv2.imread(filename)
                        newimage = cv2.resize(oriimage,(img_w,img_h))
                        newimage = 	cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
                        cv2.imshow("original image",oriimage)
                        cv2.imshow("resize image",newimage)
                        cv2.imwrite(filename, newimage)

def main():
    _resize_image(IMAGE_SUB_DIR, 28, 28)

if __name__ == "__main__":
    main()