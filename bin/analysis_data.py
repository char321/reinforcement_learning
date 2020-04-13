import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

## Get the distribution of common items and personal items
# number of common items and number of basket labels
baskets_ids = baskets_categories['bc_id']
common_items_ids = items[items['i_stock_personal'] == 'stock']['i_id']
baskets_num = baskets_ids.shape[0]
common_items_num = common_items_ids.shape[0]

participants_num = 30
items_num = 21
columns_names = ['basket' + str(id) for id in list(baskets_ids)]
stock_columns_names = [baskets_categories[baskets_categories['bc_id'] == i]['bc_label'].values[0] for i in
                       list(baskets_ids)]
stock_index_names = ['Item ' + str(id) for id in list(common_items_ids)]
stock_distribution = pd.DataFrame(np.zeros((common_items_num, baskets_num)), index=stock_index_names,
                                  columns=stock_columns_names)
index_names = stock_index_names + ['personal' + str(id) for id in list(range(1, items_num - common_items_num + 1))]
distribution = pd.DataFrame(np.zeros((items_num, baskets_num)), index=index_names, columns=columns_names)

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
        personal_i_id = (i_id - 1 - common_items_num) % (items_num - common_items_num)  # (i_id - 17) / 5
        distribution.iloc[personal_i_id + common_items_num, bc_id_1] += 1
        # distribution.iloc[personal_i_id + common_items_num, bc_id_2] += 1
    else:
        distribution.iloc[i_id - 1, bc_id_1] += 1
        stock_distribution.iloc[i_id - 1, bc_id_1] += 1
        stock_distribution.iloc[i_id - 1, bc_id_2] += 1
    count += 1

print(count == 30 * 5 + 16 * 30 - 1)
# stock_distribution = stock_distribution.drop(['basket0'], axis=1)
stock_distribution = stock_distribution.drop(['none'], axis=1)
distribution = distribution.drop(['basket0'], axis=1)
# print(stock_distribution)

stock_distribution.loc['Mean'] = stock_distribution.mean(0).T
stock_distribution['Total'] = stock_distribution.sum(1).T
print(stock_distribution)

participants_columns_names = ['P ' + str(id) for id in range(1, 31)]
participants_distributions = pd.DataFrame(np.zeros((len(columns_names), participants_num)), index=columns_names,
                                          columns=participants_columns_names)
count = 0
for row in sorts.iterrows():
    row = row[1]
    i_id = int(row['i_id'])
    p_id = int(row['p_id'])
    b_id = row['b_id']

    baskets_row = baskets[baskets['b_id'] == b_id]
    bc_id_1 = baskets_row['bc_id_1']

    if i_id > common_items_num:
        continue
    else:
        if p_id != 0:
            participants_distributions.iloc[bc_id_1, p_id - 1] += 1
    count += 1

participants_distributions['Total'] = participants_distributions.sum(1).T
participants_distributions.loc['Total'] = stock_distribution.sum(0).T

print(participants_distributions)
# print(len(stock_index_names))
# fig, ax = plt.figure()
#
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# plt.table(cellText=stock_distribution.values, colLabels=stock_distribution.columns,
#           rowLabels=stock_index_names + ['mean'], loc='top')
# plt.savefig(base_path + "/output/stock_distribution.svg")
# plt.close()

# csv_filename = base_path + "/output/total_distribution.csv"
# distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

csv_filename = base_path + "/output/stock_distribution.csv"
stock_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

## extract the common combination of the labels
dict_bc_id = {}
dict_bc_label = {}
for row in sorts.iterrows():
    row = row[1]
    i_id = int(row['i_id'])
    b_id = row['b_id']

    baskets_row = baskets[baskets['b_id'] == b_id]
    bc_id_1 = baskets_row['bc_id_1'].values[0]
    bc_id_2 = baskets_row['bc_id_2'].values[0]

    if i_id > common_items_num:
        continue
    if bc_id_2 != 0:
        label_1 = baskets_categories.iloc[bc_id_1]['bc_label']
        label_2 = baskets_categories.iloc[bc_id_2]['bc_label']

        key_id = str(bc_id_1) + '-' + str(bc_id_2) if bc_id_1 < bc_id_2 else str(bc_id_2) + '-' + str(bc_id_1)
        dict_bc_id[key_id] = dict_bc_id[key_id] + 1 if key_id in dict_bc_id else 1

        key_label = label_1 + ' & ' + label_2 if bc_id_1 < bc_id_2 else label_2 + ' & ' + label_1
        dict_bc_label[key_label] = dict_bc_label[key_label] + 1 if key_label in dict_bc_label else 1

print(sorted(dict_bc_id.items(), key=lambda d: d[1], reverse=True))
combination = sorted(dict_bc_label.items(), key=lambda d: d[1], reverse=True)
csv_filename = base_path + "/output/combination.csv"
combination = pd.DataFrame(combination)
combination.to_csv(csv_filename, float_format='%.3f', index=True, header=True)
print(sorted(dict_bc_label.items(), key=lambda d: d[1], reverse=True))


def get_user_dsitrubtion(label):
    index_names = [str(id) for id in range(1, participants_num + 1)]
    columns_names = ['Count', 'If_use']
    user_distribution = pd.DataFrame(np.zeros((participants_num, 2)), index=index_names, columns=columns_names)
    for row in sorts.iterrows():
        row = row[1]
        if (not str.isdigit(str(row['i_id']))) or (not str.isdigit(str(row['b_id']))) or (
                not str.isdigit(str(row['b_id_alt']))):
            continue

        i_id = int(row['i_id'])
        b_id = row['b_id']
        p_id = row['p_id']

        baskets_row = baskets[baskets['b_id'] == b_id]
        bc_id_1 = baskets_row['bc_id_1']
        bc_id_2 = baskets_row['bc_id_2']

        if i_id > common_items_num:
            continue
        if label == int(bc_id_1):
            user_distribution.iloc[p_id - 1, 0] += 1
            user_distribution.iloc[p_id - 1, 1] = 1
        if label == int(bc_id_2):
            user_distribution.iloc[p_id - 1, 0] += 1
            user_distribution.iloc[p_id - 1, 1] = 1

    # user_distribution.loc['Total'] = user_distribution.sum(0).T

    return user_distribution


for i in [1, 2]:
    user_distribution = get_user_dsitrubtion(i)

    csv_filename = base_path + "/output/user_distribution" + str(i) + ".csv"
    user_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

    a = plt.figure(i, figsize=(4.5, 3))
    plt.bar(x=user_distribution.index, height=user_distribution['Count'])
    plt.tick_params(labelsize=5)
    plt.xlabel('Participant ID', fontsize=8)
    plt.ylabel('Usage Count', fontsize=8)
    plt.ylim((0, 16))
    plt.yticks(np.arange(0, 16, 2))

    t = baskets_categories[baskets_categories['bc_id'] == i]['bc_label']

    plt.title('Usage Distribution of Label `' + str(t.values[0]) + '`', fontsize=10)
    plt.savefig(base_path + "/output/user_distribution" + str(i) + '.svg')
    plt.close(a)

for i in [3, 10]:
    user_distribution = get_user_dsitrubtion(i)

    csv_filename = base_path + "/output/user_distribution" + str(i) + ".csv"
    user_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

    a = plt.figure(i, figsize=(4.5, 3))
    plt.bar(x=user_distribution.index, height=user_distribution['Count'])
    plt.tick_params(labelsize=5)
    plt.xlabel('Participant ID', fontsize=8)
    plt.ylabel('Usage Count', fontsize=8)
    plt.ylim((0, 18))
    plt.yticks(np.arange(0, 18, 2))

    t = baskets_categories[baskets_categories['bc_id'] == i]['bc_label']

    plt.title('Usage Distribution of Label `' + str(t.values[0]) + '`', fontsize=10)
    plt.savefig(base_path + "/output/user_distribution" + str(i) + '.svg')
    plt.close(a)

for i in [6, 7, 8, 9, 11]:
    user_distribution = get_user_dsitrubtion(i)

    csv_filename = base_path + "/output/user_distribution" + str(i) + ".csv"
    user_distribution.to_csv(csv_filename, float_format='%.3f', index=True, header=True)

    a = plt.figure(i, figsize=(4.5, 3))
    plt.bar(x=user_distribution.index, height=user_distribution['Count'])
    plt.tick_params(labelsize=5)
    plt.xlabel('Participant ID', fontsize=8)
    plt.ylabel('Usage Count', fontsize=8)
    plt.ylim((0, 22))
    plt.yticks(np.arange(0, 22, 2))

    t = baskets_categories[baskets_categories['bc_id'] == i]['bc_label']

    plt.title('Usage Distribution of Label `' + str(t.values[0]) + '`', fontsize=10)
    plt.savefig(base_path + "/output/user_distribution" + str(i) + '.svg')
    plt.close(a)

# 1+2: white and light: most common combination
# 3+5, 3+7, 3+9, 3+10, 3+11: colour + description
# 2+8, 2+9, 2+10: colour + description
# 1+6, 1+8: colour + description
# 7+9, 6+7, 6+8, 5+10: description + description
# Consider replaceable of the baskets
# 2, 3 are more common to combined with other label
# choose 1, 3, 5 # - more replaceable
# 5: majority + wider range
# 1: 1 and 2 is similar, but 1 is more specific
# 3: majority + wider range
# overall: based on colour, can cover most clothes, easy to extend: add new basket with specific purpose such as mix, children
# show in tables # - may assist with graph (data visualization)
# calculate the proportion # - algorithm to choose the labels? -- 量化
