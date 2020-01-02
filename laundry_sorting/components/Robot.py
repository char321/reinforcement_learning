class Robot:
    def __init__(self):
        self.full_labels =  {1: 'white', 2: 'light', 3: 'dark', 4: 'bright', 5: 'colour',
                        6: 'handwash', 7: 'denim', 8: 'delicate', 9: 'children', 10: 'mixed', 11: 'miscellaneous'}

    def pick(self, i_id, i_colour, i_type):
        print('Pick up items %d which is %s %s...' % (i_id, i_colour, i_type))

    def put(self, b_id):
        print('Put into basket %s' % b_id)

    def moving(self, b_id):
        print('Moving to basket %s' % b_id)

    def add_new_label(self, label_id, baskets):
        baskets[label_id] = self.full_labels[label_id]
        return baskets