import pandas as pd
import numpy as np
import pprint
from models.QLearning import QLearning
from models.QLearningModel import QLearningModel


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

    def pick(self, i_id, i_colour, i_type):
        print('Pick up items %d which is %s %s...' % (i_id, i_colour, i_type))

    def put(self, b_id):
        print('Put into basket %s' % b_id)

    def moving(self, b_id):
        print('Moving to basket %s' % b_id)

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

    def ask_for_label(self):
        # TODO
        print('Input label: ')


class Controller:
    def __init__(self):
        self.robot = Robot()
        self.data = load_data()
        # self.baskets = {1: 'white', 3: 'dark', 5: 'colour'}
        self.baskets = {1: 'whites', 2: 'lights', 3: 'darks', 4: 'brights', 5: 'colours',
                        6: 'handwash', 7: 'denims', 8: 'delicates', 9: 'children', 10: 'mixed', 11: 'miscellaneous'}
        self.nob = len(self.baskets)  # number of baskets
        self.mob = 6  # max number of baskets
        self.model = QLearningModel(self.nob)

    def train(self):
        gamma = 0.8
        noi = 5000  # number of iterations
        print('Training...')

        self.model.train(gamma, noi, self.data, self.baskets, self.robot)

    def test_person(self, p_id):
        q_table = self.model.get_q_table()
        # print(q_table)

        clothes = self.data[p_id]
        results = {}
        for i_id in clothes.keys():
            cloth = clothes[i_id]
            correct_label = [cloth['bc_id_1'], cloth['bc_id_2']]

            # Get the information of cloth
            i_colour = self.robot.map_to_colour(cloth['i_colour'])
            i_type = self.robot.map_to_type(cloth['i_type'])
            colour_index = self.model.colours.index(i_colour)
            type_index = self.model.types.index(i_type)
            state = colour_index * len(self.model.types) + type_index

            actions = q_table[state]
            action = np.argmax(actions)
            label = list(self.baskets.keys())[action]

            # print(label)
            # print(correct_label)
            result = 1 if label in correct_label else 0
            results[i_id] = result

        print("Person %s" % str(p_id))
        print(results)
        return results

    def test_all(self):
        total_accuracy = 0
        for p_id in range(1, 31):
            results = self.test_person(p_id)
            total_accuracy += (sum(results.values()) / 16) / 30

        print(total_accuracy)

    def update(self, p_id):
        gamma = 0.8
        noi = 5000  # number of iterations
        print('Updating...')

        self.model.update(gamma, noi, self.data, self.baskets, self.robot, p_id)

controller = Controller()
controller.train()
controller.test_all()

q_table = np.copy(controller.model.get_q_table())
total_accuracy = 0
for p_id in range(1, 31):
    controller.model.set_q_table(np.copy(q_table))
    controller.update(p_id)
    results = controller.test_person(p_id)
    total_accuracy += (sum(results.values()) / 16) / 30
print(total_accuracy)


