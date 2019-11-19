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

baskets_ids = baskets_categories['bc_id']
common_items_ids = items[items['i_stock_personal'] == 'stock']['i_id']
baskets_num = baskets_ids.shape[0]
common_items_num = common_items_ids.shape[0]

columns_names = ['basket' + str(id) for id in list(baskets_ids)]
index_names = ['item' + str(id) for id in list(common_items_ids)]

# calculate the distrubution
distribution = pd.DataFrame(np.zeros((common_items_num, baskets_num)), index=index_names, columns=columns_names)

# for i in range(common_items_num):
#     items = sorts[sorts['i_id'] == str(i + 1)]
#     for row in items.iterrows():
#         print(row[1]['b_id'])

print(set(sorts['b_id_alt']))
for row in sorts.iterrows():
    i_id = int(row[1]['i_id'])
    # TODO link b_id to bc_id
    b_id = int(row[1]['b_id'])
    b_id_alt = int(row[1]['b_id_alt'])
    if i_id > common_items_num:
        continue
    print(i_id)
    print(b_id)
    distribution.iloc[i_id - 1, b_id] += 1
    distribution.iloc[i_id - 1, b_id_alt] += 1

print(distribution)


