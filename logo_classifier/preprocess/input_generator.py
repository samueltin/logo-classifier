import os
import random
import cv2
import numpy as np


class InputGenerator(object):
    __slots__ = ['image_path', 'logos', 'image_dict', 'image_size']

    def __init__(self, image_path, image_size, sub_folders = None ):
        self.image_path = image_path
        self.logos = sub_folders if sub_folders else []
        self.image_dict = {}
        self.image_size = image_size
        for logo in self.logos:
            image_list = os.listdir(os.path.join(self.image_path, logo))
            if '.DS_Store' in image_list:
                image_list.remove('.DS_Store')
            self.image_dict[logo]=image_list

    def get_logos(self):
        return list(self.logos)

    def input_gen(self):
        logo_idx = random.choice(range(0, len(self.logos) - 1, 1))
        image_list = self.image_dict[self.logos[logo_idx]]
        image_idx = random.choice(range(0, len(image_list) - 1, 1))

        filename = os.join.path(self.image_path, image_list[image_idx])

        image = cv2.imread(filename)
        image = cv2.resize(image, (self.image_size, self.image_size), cv2.INTER_LINEAR)
        image = np.array(image, dtype=np.uint8)
        image = image.astype('float32')
        image = np.multiply(image, 1.0 / 255.0)

        one_hot = np.zeros(len(self.logos))
        one_hot[image_idx] = 1

        return image, one_hot


