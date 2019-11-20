import pandas as pd
import numpy as np

file_name = 'Database.xlsx'

# read data
baskets = pd.read_excel(file_name, sheet_name='baskets')
baskets_categories = pd.read_excel(file_name, sheet_name='baskets_categories')
items = pd.read_excel(file_name, sheet_name='items')
items_stock = pd.read_excel(file_name, sheet_name='items_stock')
participants = pd.read_excel(file_name, sheet_name='participants')
sorts = pd.read_excel(file_name, sheet_name='sorts')

# number of common items and number of basket labels
baskets_ids = baskets_categories['bc_id']
common_items_ids = items[items['i_stock_personal'] == 'stock']['i_id']
baskets_num = baskets_ids.shape[0]
common_items_num = common_items_ids.shape[0]

columns_names = ['basket' + str(id) for id in list(baskets_ids)]
index_names = ['item' + str(id) for id in list(common_items_ids)]

# calculate the common items distribution
common_distribution = pd.DataFrame(np.zeros((common_items_num, baskets_num)), index=index_names, columns=columns_names)

# for i in range(common_items_num):
#     items = sorts[sorts['i_id'] == str(i + 1)]
#     for row in items.iterrows():
#         print(row[1]['b_id'])

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

    if i_id > common_items_num:
        continue

    baskets_row = baskets[baskets['b_id'] == b_id]
    bc_id_1 = baskets_row['bc_id_1']
    bc_id_2 = baskets_row['bc_id_2']

    common_distribution.iloc[i_id - 1, bc_id_1] += 1
    common_distribution.iloc[i_id - 1, bc_id_2] += 1
    count += 1

print(count == 16 * 30 - 1)  # check the all items have been counted
print(common_distribution)

## TODO
# Choose basket label
# - majority
# - which label cover wider range
# - which is more replaceable
# - may assist with graph (data visualization)

# Extend to all the items
