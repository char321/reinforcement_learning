import pandas as pd
import numpy as np
import os
from IPython.display import display

dir_abs_name = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(dir_abs_name) + '/data'
file_name = '/database/Database.xlsx'

# read data
data_path = base_path + file_name
baskets = pd.read_excel(data_path, sheet_name='baskets')
baskets_categories = pd.read_excel(data_path, sheet_name='baskets_categories')
items = pd.read_excel(data_path, sheet_name='items')
items_stock = pd.read_excel(data_path, sheet_name='items_stock')
participants = pd.read_excel(data_path, sheet_name='participants')
sorts = pd.read_excel(data_path, sheet_name='sorts')


def map_to_colour(i_colour):
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


def map_to_type(i_type):
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


def isnumber(aString):
    try:
        float(aString)
        return True
    except:
        return False


types = set(sorts['s_label'])
for i_type in types:
    if isnumber(i_type):
        continue
    if map_to_type(i_type) == 'others':
        print(i_type)

colours = set(sorts['s_colour_description'])
for colour in colours:
    if isnumber(colour):
        continue
    if map_to_colour(colour) == 'colours':
        print(colour)
