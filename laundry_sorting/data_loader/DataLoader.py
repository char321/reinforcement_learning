import pandas as pd
import os
import pprint


class DataLoader:
    def __init__(self):
        dir_abs_name = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_abs_name)
        base_path = os.path.dirname(base_path) + '/data'
        file_name = '/database/Database.xlsx'

        # read data
        data_path = base_path + file_name
        self.baskets = pd.read_excel(data_path, sheet_name='baskets')
        self.baskets_categories = pd.read_excel(data_path, sheet_name='baskets_categories')
        self.items = pd.read_excel(data_path, sheet_name='items')
        self.items_stock = pd.read_excel(data_path, sheet_name='items_stock')
        self.sorts = pd.read_excel(data_path, sheet_name='sorts')

    def map_to_colour(self, i_colour):
        if 'white' in i_colour:
            return 'white'
        elif 'black' in i_colour:
            return 'black'
        elif 'dark' in i_colour:
            return 'dark'
        else:
            return 'colours'

    def map_to_type(self, i_type):
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
                    i_colour = self.map_to_colour(item['is_colour'].values[0])
                    i_type = self.map_to_type(item['is_label'].values[0])
                    # print(i_colour)
                    # print(i_type)
                    basket = self.baskets[self.baskets['b_id'] == int(b_id)]
                    bc_id_1 = basket['bc_id_1'].values[0]
                    bc_id_2 = basket['bc_id_2'].values[0]
                    b_label = basket['b_label'].values[0]

                    cloth = {}
                    cloth['i_colour'] = i_colour
                    cloth['i_type'] = i_type
                    cloth['b_id'] = b_id
                    cloth['bc_id_1'] = bc_id_1
                    cloth['bc_id_2'] = bc_id_2
                    cloth['b_label'] = b_label

                    clothes[int(i_id)] = cloth

            persons[p_id] = clothes

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(persons)

        return persons

    def load_baskets_categories(self):
        # TODO
        return
