import pandas as pd
import numpy as np
from IPython.display import display

file_name = 'Database.xlsx'

# read data
baskets = pd.read_excel(file_name, sheet_name='baskets')
baskets_categories = pd.read_excel(file_name, sheet_name='baskets_categories')
items = pd.read_excel(file_name, sheet_name='items')
items_stock = pd.read_excel(file_name, sheet_name='items_stock')
participants = pd.read_excel(file_name, sheet_name='participants')
sorts = pd.read_excel(file_name, sheet_name='sorts')

## Get the distribution of common items and personal items
# number of common items and number of basket labels
baskets_ids = baskets_categories['bc_id']
common_items_ids = items[items['i_stock_personal'] == 'stock']['i_id']
baskets_num = baskets_ids.shape[0]
common_items_num = common_items_ids.shape[0]

items_num = 21
columns_names = ['basket' + str(id) for id in list(baskets_ids)]
index_names = ['common_item' + str(id) for id in list(common_items_ids)]
index_names += ['personal_item' + str(id) for id in list(range(1, items_num - common_items_num + 1))]
total_distribution = pd.DataFrame(np.zeros((items_num, baskets_num)), index=index_names, columns=columns_names)

count = 0
for row in sorts.iterrows():
    row = row[1]
    if (not str.isdigit(str(row['i_id']))) or (not str.isdigit(str(row['b_id']))) or (
            not str.isdigit(str(row['b_id_alt']))):
        print(row)
        continue

    i_id = int(row['i_id'])
    b_id = row['b_id']
    b_id_alt = row['b_id_alt']

    baskets_row = baskets[baskets['b_id'] == b_id]
    bc_id_1 = baskets_row['bc_id_1']
    bc_id_2 = baskets_row['bc_id_2']

    if i_id > common_items_num:
        personal_i_id = (i_id - 1 - common_items_num) % (items_num - common_items_num) # (i_id - 17) / 5
        total_distribution.iloc[personal_i_id + common_items_num, bc_id_1] += 1
        # total_distribution.iloc[personal_i_id + common_items_num, bc_id_2] += 1
    else:
        total_distribution.iloc[i_id - 1, bc_id_1] += 1
        # total_distribution.iloc[i_id - 1, bc_id_2] += 1
    count += 1

print(count == 30 * 5 + 16 * 30 - 1)
print(total_distribution)

# csv_filename = "total_distribution.csv"
# total_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

csv_filename = "first_choice_distribution.csv"
total_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

## extract the common combination of the labels
dict = {}
for row in baskets.iterrows():
    row = row[1]
    bc_id_1 = int(row['bc_id_1'])
    bc_id_2 = int(row['bc_id_2'])
    if bc_id_2 != 0:
        key = str(bc_id_1) + '-' + str(bc_id_2) if bc_id_1 < bc_id_2 else str(bc_id_2) + '-' + str(bc_id_1)
        dict[key] = dict[key] + 1 if key in dict else 1

print(sorted(dict.items(), key=lambda d: d[1], reverse=True))
# 1+2: white and light: most common combination
# 3+5, 3+7, 3+9, 3+10, 3+11: colour + description
# 2+8, 2+9, 2+10: colour + description
# 1+6, 1+8: colour + description
# 7+9, 6+7, 6+8, 5+10: description + description
# Consider replaceable of the baskets
# 2, 3 are more common to combined with other label
# choose 1, 3, 5
# 5: majority + wider range
# 1: 1 and 2 is similar, but 1 is more specific
# 3: majority + wider range
# overall: based on colour, can cover most clothes, easy to extend: add new basket with specific purpose such as mix, children

## TODO
# Choose basket label
# - majority
# - which label cover wider range
# - which is more replaceable
# - may assist with graph (data visualization)
# - decision tree?
# - algorithm to devide? -- 量化

# Extend to all the items
