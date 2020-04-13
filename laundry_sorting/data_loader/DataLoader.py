import os
import cv2
import pandas as pd
import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa


class DataLoader:
    def __init__(self):
        dir_abs_name = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_abs_name)
        self.base_path = os.path.dirname(base_path) + '/data'
        file_name = '/database/Database.xlsx'

        # read excel file
        data_path = self.base_path + file_name
        self.baskets = pd.read_excel(data_path, sheet_name='baskets')
        self.baskets_categories = pd.read_excel(data_path, sheet_name='baskets_categories')
        self.items = pd.read_excel(data_path, sheet_name='items')
        self.items_stock = pd.read_excel(data_path, sheet_name='items_stock')
        self.sorts = pd.read_excel(data_path, sheet_name='sorts')

        self.cloth_data = None  # cloth sorting data
        self.image_data = None  # image data without image augmentation
        self.imgaug_data = None  # image data with image augmentation

    def isnumber(self, aString):
        try:
            float(aString)
            return True
        except:
            return False

    def get_colours(self):
        # simple
        # return ['white', 'black', 'dark', 'colours']

        # full
        return ['white', 'black', 'dark', 'light', 'bright', 'colours']

    def get_types(self):
        # simple
        # return ['t-shirt', 'socks', 'polo', 'pants', 'jeans', 'shirt', 'skirt', 'others']

        # full
        return ['t-shirt', 'sport', 'top', 'socks', 'polo', 'pants', 'jeans', 'shorts', 'shirt', 'skirt', 'pyjama',
                'hat', 'baby', 'others']

    # simple colour mapping
    def map_to_colour_simple(self, i_colour):
        if self.isnumber(i_colour):
            return 'colours'

        # TODO - grey
        if 'white' in i_colour:
            return 'white'
        elif 'black' in i_colour:
            return 'black'
        elif 'dark' in i_colour:
            return 'dark'
        else:
            return 'colours'

    # simple type mapping
    def map_to_type_simple(self, i_type):
        if self.isnumber(i_type):
            return 'others'

        if 't-shirt' in i_type:
            return 't-shirt'
        elif 'socks' in i_type:
            return 'socks'
        elif 'polo' in i_type:
            return 'polo'
        elif 'pants' in i_type or 'jogger' in i_type:
            return 'pants'
        elif 'jeans' in i_type:
            return 'jeans'
        elif 'shirt' in i_type or 'blouse' in i_type:
            return 'shirt'
        elif 'skirt' in i_type:
            return 'skirt'
        else:
            return 'others'

    # complex colour mapping
    def map_to_colour_full(self, i_colour):
        if self.isnumber(i_colour):
            return 'colours'

        # TODO - grey
        if 'white' in i_colour:
            return 'white'
        elif 'black' in i_colour:
            return 'black'
        elif 'dark' in i_colour:
            return 'dark'
        elif 'light' in i_colour:
            return 'light'
        elif 'bright' in i_colour:
            return 'bright'
        else:
            return 'colours'

    # complex type mapping
    def map_to_type_full(self, i_type):
        if self.isnumber(i_type):
            return 'others'

        if 't-shirt' in i_type or 'tee' in i_type:
            return 't-shirt'
        elif 'sport' in i_type or 'swimming' in i_type or 'running' in i_type or 'gym' in i_type or 'football' in i_type or 'fitness' in i_type or 'rugby' in i_type or 'athletic' in i_type or 'boxers' in i_type or 'legging' in i_type:
            return 'sport'
        elif 'vest' in i_type or 'hoodie' in i_type or 'long_sleeve' in i_type or 'sweater' in i_type or 'neck' in i_type or 'top' in i_type or 'jumper' in i_type or 'base layer' in i_type:
            return 'top'
        elif 'socks' in i_type or 'sock' in i_type:
            return 'socks'
        elif 'polo' in i_type:
            return 'polo'
        elif 'pants' in i_type or 'pant' in i_type or 'jogger' in i_type or 'bottoms' in i_type or 'trousers' in i_type:
            return 'pants'
        elif 'jeans' in i_type:
            return 'jeans'
        elif 'shorts' in i_type:
            return 'shorts'
        elif 'shirt' in i_type or 'blouse' in i_type:
            return 'shirt'
        elif 'skirt' in i_type or 'dress' in i_type:
            return 'skirt'
        elif 'pyjama' in i_type:
            return 'pyjama'
        elif 'beanie' in i_type or 'hat' in i_type or 'balaclava' in i_type or 'headband' in i_type:
            return 'hat'
        elif 'baby' in i_type:
            return 'baby'
        else:
            return 'others'

    # load stock items sorting data
    def load_stock_data(self):
        # generate person dict
        persons_clothes = {}
        p_ids = set(self.sorts['p_id'])
        for p_id in p_ids:
            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']
            clothes = {}
            for i_id in i_ids:
                # focus on stock items first
                if int(i_id) <= 16:
                    item = self.items_stock[self.items_stock['i_id'] == int(i_id)]
                    b_id = temp_sorts['b_id'][temp_sorts['i_id'] == i_id].values[0]
                    i_colour = self.map_to_colour_full(item['is_colour'].values[0])
                    i_type = self.map_to_type_full(item['is_label'].values[0])

                    basket = self.baskets[self.baskets['b_id'] == int(b_id)]
                    bc_id_1 = basket['bc_id_1'].values[0]
                    bc_id_2 = basket['bc_id_2'].values[0]

                    cloth = {'i_colour': i_colour, 'i_type': i_type, 'b_id': b_id, 'bc_id_1': bc_id_1,
                             'bc_id_2': bc_id_2}
                    clothes[int(i_id)] = cloth
            persons_clothes[p_id] = clothes
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(persons)
        self.cloth_data = persons_clothes
        return persons_clothes

    # load all the sorting data
    def load_all_data(self):
        persons_clothes = {}
        p_ids = set(self.sorts['p_id'])
        for p_id in p_ids:
            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']
            clothes = {}
            for i_id in i_ids:
                sort = temp_sorts[temp_sorts['i_id'] == i_id]  # item sorting info
                b_id = temp_sorts['b_id'][temp_sorts['i_id'] == i_id].values[0]

                i_colour = self.map_to_colour_full(sort['s_colour_description'].values[0])
                i_type = self.map_to_type_simple(sort['s_label'].values[0])

                basket = self.baskets[self.baskets['b_id'] == int(b_id)]
                bc_id_1 = basket['bc_id_1'].values[0]
                bc_id_2 = basket['bc_id_2'].values[0]

                cloth = {'i_colour': i_colour, 'i_type': i_type, 'b_id': b_id, 'bc_id_1': bc_id_1,
                         'bc_id_2': bc_id_2}
                clothes[int(i_id)] = cloth

            persons_clothes[p_id] = clothes
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(persons)
        self.cloth_data = persons_clothes
        return persons_clothes

    # use to resize original images
    # call only once
    def resize_image(self):
        p_ids = set(self.sorts['p_id'])

        for p_id in p_ids:
            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']

            for i_id in i_ids:
                print(i_id)
                item = self.items[self.items['i_id'] == int(i_id)]
                image_name = str(item['i_image_front'].values[0]) + '.jpg'
                if 'Photo' in image_name:
                    image_name = image_name.replace('/Photo/', '/')

                image_path = self.base_path + '/images/' + image_name

                try:
                    img = Image.open(image_path)
                except:
                    continue

                res = img.resize((300, 400), Image.ANTIALIAS)

                new_image_path = self.base_path + '/new_images/img' + str(i_id) + '.jpg'
                res.save(new_image_path)

    # load resized images without image augmentation
    def load_new_images(self):
        persons_images = {}
        p_ids = set(self.sorts['p_id'])
        n = {}

        for p_id in p_ids:
            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']
            images = {}

            for i_id in i_ids:
                image_path = self.base_path + '/new_images/img' + str(i_id) + '.jpg'
                img = cv2.imread(image_path)

                if img is None:
                    n[str(i_id)] = image_path
                else:
                    images[int(i_id)] = img

            persons_images[p_id] = images
        self.image_data = persons_images
        return persons_images

    # load images and do the image augmentation to generate more data
    def image_aug(self, isCommon=False):
        aug_images = {}
        p_ids = set(self.sorts['p_id'])
        n = {}

        for p_id in p_ids:
            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']
            images = []

            for i_id in i_ids:
                if isCommon and i_id > 16:
                    break
                image_path = self.base_path + '/new_images/img' + str(i_id) + '.jpg'
                orig = cv2.imread(image_path)

                if orig is None:
                    n[str(i_id)] = image_path
                    continue

                # Image Augmentation
                ia.seed(3)
                seq1 = iaa.Sequential([
                    iaa.Flipud(0.5),
                    iaa.Fliplr(0.5),
                    iaa.GaussianBlur(sigma=(0.0, 1.5)),
                    iaa.Add((-20, 20))
                ])
                seq2 = iaa.Sequential([
                    iaa.Flipud(0.5),
                    iaa.Fliplr(0.5),
                    iaa.ShearY((-20, 20)),
                    iaa.Add((-20, 20))
                ])
                seq3 = iaa.Sequential([
                    iaa.Flipud(0.5),
                    iaa.Fliplr(0.5),
                    iaa.Rotate((-180, 180)),
                    iaa.PerspectiveTransform(scale=(0.01, 0.1))
                ])

                img_list = {
                    'og': orig,
                    'ud': iaa.Flipud(1.0).augment_image(orig),
                    'lr': iaa.Fliplr(1.0).augment_image(orig),
                    'affine': iaa.ShearY((-20, 20)).augment_image(orig),
                    'rot1': iaa.Rotate((0, 90)).augment_image(orig),
                    'rot2': iaa.Rotate((90, 180)).augment_image(orig),
                    'scale': iaa.PerspectiveTransform(scale=(0.01, 0.15)).augment_image(orig),
                    'blur': iaa.GaussianBlur(sigma=(0.0, 1.5)).augment_image(orig),
                    'add': iaa.Add((-20, 20)).augment_image(orig),
                    'com1': seq1.augment_image(orig),
                    'com2': seq2.augment_image(orig),
                    'com3': seq3.augment_image(orig)
                }

                clothes = self.cloth_data[p_id]
                cloth = clothes[i_id]
                correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

                for k, v in img_list.items():
                    img_type = k
                    img_data = v
                    img = {
                        'p_id': p_id,
                        'i_id': i_id,
                        'type': img_type,
                        'data': img_data,
                        'label': correct_label
                    }
                    images.append(img)

            aug_images[p_id] = images
            # n[p_id] = temp
        self.imgaug_data = aug_images
        return aug_images

    def load_baskets_categories(self):
        # TODO
        return

