import pandas as pd
import pprint

class Database:

    def load_data(self):
        file_name = 'Database.xlsx'

        # read data
        baskets = pd.read_excel(file_name, sheet_name='baskets')
        baskets_categories = pd.read_excel(file_name, sheet_name='baskets_categories')
        items = pd.read_excel(file_name, sheet_name='items')
        items_stock = pd.read_excel(file_name, sheet_name='items_stock')
        participants = pd.read_excel(file_name, sheet_name='participants')
        sorts = pd.read_excel(file_name, sheet_name='sorts')

        # generate person dict
        persons = {}
        p_ids = set(sorts['p_id'])
        for p_id in p_ids:

            temp_sorts = sorts[sorts['p_id'] == p_id]
            i_ids = temp_sorts['i_id']

            clothes = {}
            for i_id in i_ids:
                # focus on stock items first
                if int(i_id) <= 16:
                    item = items_stock[items_stock['i_id'] == int(i_id)]
                    sort = temp_sorts[temp_sorts['i_id'] == i_id]
                    b_id = temp_sorts['b_id'][temp_sorts['i_id'] == i_id].values[0]
                    # print(i_id)
                    i_colour = item['is_colour'].values[0]
                    i_type = item['is_label'].values[0]
                    # print(i_colour)
                    # print(i_type)
                    basket = baskets[baskets['b_id'] == int(b_id)]
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
