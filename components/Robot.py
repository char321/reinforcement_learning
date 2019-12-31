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

    def add_new_label(self, label_id, baskets):
        # TODO - check if label is already existed
        baskets[label_id] = self.full_labels[label_id]
        return baskets