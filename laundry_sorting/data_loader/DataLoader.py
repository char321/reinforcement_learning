import pandas as pd
import os
import cv2
import pprint


class DataLoader:
    def __init__(self):
        dir_abs_name = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_abs_name)
        self.base_path = os.path.dirname(base_path) + '/data'
        file_name = '/database/Database.xlsx'

        # read data
        data_path = self.base_path + file_name
        self.baskets = pd.read_excel(data_path, sheet_name='baskets')
        self.baskets_categories = pd.read_excel(data_path, sheet_name='baskets_categories')
        self.items = pd.read_excel(data_path, sheet_name='items')
        self.items_stock = pd.read_excel(data_path, sheet_name='items_stock')
        self.sorts = pd.read_excel(data_path, sheet_name='sorts')

    def isnumber(self, aString):
        try:
            float(aString)
            return True
        except:
            return False

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

    def load_sorting_data(self):
        # generate person dict
        persons = {}
        p_ids = set(self.sorts['p_id'])
        for p_id in p_ids:

            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']

            clothes = {}
            for i_id in i_ids:
                # focus on stock items first
                if int(i_id) <= 16:
                    item = self.items_stock[self.items_stock['i_id'] == int(i_id)]
                    sort = temp_sorts[temp_sorts['i_id'] == i_id]
                    b_id = temp_sorts['b_id'][temp_sorts['i_id'] == i_id].values[0]
                    # print(i_id)
                    i_colour = self.map_to_colour_full(item['is_colour'].values[0])
                    i_type = self.map_to_type_full(item['is_label'].values[0])
                    # print(i_colour)
                    # print(i_type)
                    basket = self.baskets[self.baskets['b_id'] == int(b_id)]
                    bc_id_1 = basket['bc_id_1'].values[0]
                    bc_id_2 = basket['bc_id_2'].values[0]
                    b_label = basket['b_label'].values[0]

                    cloth = {'i_colour': i_colour, 'i_type': i_type, 'b_id': b_id, 'bc_id_1': bc_id_1,
                             'bc_id_2': bc_id_2, 'b_label': b_label}

                    clothes[int(i_id)] = cloth

            persons[p_id] = clothes

        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(persons)

        return persons

    def load_all_data(self):
        persons = {}
        p_ids = set(self.sorts['p_id'])
        for p_id in p_ids:

            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']

            clothes = {}
            for i_id in i_ids:
                # focus on stock items first
                item = self.items_stock[self.items_stock['i_id'] == int(i_id)]
                sort = temp_sorts[temp_sorts['i_id'] == i_id]
                b_id = temp_sorts['b_id'][temp_sorts['i_id'] == i_id].values[0]
                # print(i_id)
                i_colour = self.map_to_colour_full(sort['s_colour_description'].values[0])
                i_type = self.map_to_type_simple(sort['s_label'].values[0])
                # print(i_colour)
                # print(i_type)
                basket = self.baskets[self.baskets['b_id'] == int(b_id)]
                bc_id_1 = basket['bc_id_1'].values[0]
                bc_id_2 = basket['bc_id_2'].values[0]
                b_label = basket['b_label'].values[0]

                cloth = {'i_colour': i_colour, 'i_type': i_type, 'b_id': b_id, 'bc_id_1': bc_id_1,
                         'bc_id_2': bc_id_2, 'b_label': b_label}

                clothes[int(i_id)] = cloth

            persons[p_id] = clothes

        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(persons)

        return persons

    # TODO - name
    def load_images(self):
        persons = {}
        p_ids = set(self.sorts['p_id'])

        n = {}
        for p_id in p_ids:

            temp_sorts = self.sorts[self.sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']

            clothes = {}
            for i_id in i_ids:
                # focus on stock items first
                item = self.items[self.items['i_id'] == int(i_id)]
                image_name = str(item['i_image_front'].values[0]) + '.jpg'
                if 'Photo' in image_name:
                    image_name = image_name.replace('/Photo/', '/')

                image_path = self.base_path + '/images/' + image_name
                img = cv2.imread(image_path)

                # print(image_name)
                if img is None:
                    n[str(i_id)] = image_path

                clothes[int(i_id)] = img

            persons[p_id] = clothes
            # n[p_id] = temp
        print(clothes)
        # print(n)
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(persons)

    def load_baskets_categories(self):
        # TODO
        return

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
