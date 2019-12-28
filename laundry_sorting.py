import pandas as pd
import pprint
from models.QLearning import QLearning


def load_data():
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

# persons
# - key: person id
# - value: clothes sorting information (as a dict)
#   clothes:
#       - key: i_id
#       - value: clothe sorting information including colour, type, basket id, basket category id, basket label

class Sensor:
    def detect(self, i_id):
        colour = ''
        type = ''
        return {colour, type}

class Robot:
    def __init__(self):
        self.sensor = Sensor()

    def pick(self, i_id):
        print('Pick up items %d...' % i_id)

    def put(self, b_id):
        print('Put into basket %i' % b_id)

    def moving(self, b_id):
        print('Moving to basket %i' % b_id)

    def ask_for_label(self):
        # TODO
        print('Input label: ')

class Controller:
    def __init__(self):
        self.robot = Robot()
        self.data = load_data()
        self.model = QLearning()
        self.baskets = {1: 'white', 3: 'dark', 5: 'colour'}
        self.nob = 3 # number of baskets
        self.mob = 6 # max number of baskets

    def train(self):
        gamma = 0.8
        noi = 1000 # number of iterations
        print('Training...')

        self.model.train(gamma, noi, self.data, self.baskets, self.robot)

        print(self.model.get_result())

controller = Controller()
controller.train()