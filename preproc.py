import re
import pandas as pd
import numpy as np

# File cleaning
# --------------------
# with open('/home/beast/Downloads/1000bartender.txt', 'r', errors='ignore') as f:
#     array = f.readlines()
# idx = array.index('1. BLOODY MARY\n')
# idx_end = array.index('Index	553\n')
# array = array[idx:idx_end-1]
#
#
# def remove_spam(file, what):
#     for _ in range(array.count(what)):
#         file.remove(what)
#
# remove_spam(array, " 1000 BEST BARTENDER'S RECIPES\n")
# remove_spam(array, '\n')
#
# with open('modif_file', 'w') as f:
#     f.write(''.join(array))
# --------------------

# Name extraction
# -----------------------
# lst = []
# for cnt in range(1, 1001):
#     for line in array:
#         if line.startswith(str(cnt)+'.'):
#             # print(line.replace('\n', ''))
#             lst.append(line.replace('\n', ''))
#             break
#         if line == array[-1]:
#             print(cnt)
#
# with open('coc_names', 'w') as f:
#     f.write('\n'.join(lst))
# -----------------------

with open('coc_names', 'r') as f, open('modif_file', 'r') as f2:
    names = f.readlines()
    array = f2.readlines()
# print(array, names, sep='\n')

dct = {}
for name1, name2 in zip(names[:], names[1:]):
    try:
        dct[name1] = array[array.index(name1)+1:array.index(name2)]
    except ValueError:
        dct[name1] = array[array.index(name1)+1:-1]
# print(dct)

dataframe = pd.DataFrame()
dataframe['cocktails'] = dct.keys()
dataframe['cocktails_extended'] = None

# Extracting extending string for cocktail name
for name in dct.keys():
    st = name + ''.join(dct[name])
    dataframe.loc[dataframe['cocktails'] == name, 'cocktails_extended'] = '\n'.join(re.findall('(^\d{1,4}?\..+)\n([A-Z]{3,100}?\n)?', st)[0])

# Extracting view name for each cocktail
dataframe['work_name'] = (dataframe['cocktails_extended']
                              .str.split(' ')
                              .str[1:].str.join(' ').apply(lambda x: ' '.join(i for i in x.split('\n') if i)))

# sort by index and adding description
dataframe['description'] = None
dataframe['idx'] = dataframe['cocktails'].str.split(' ').str[0].str.split('.').str[0].astype(np.float32)
dataframe.sort_values('idx', inplace=True)

for work_name1, work_name2 in zip(dataframe['cocktails'], dataframe['cocktails'][1:].append(pd.Series(array[-1]))):
    dataframe.loc[dataframe['cocktails'] == work_name1, 'description'] = ' '.join(array[array.index(work_name1) + 1: array.index(work_name2)])

# print(dataframe)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dataframe)

with open('DF', 'w') as f:
    dataframe.to_csv(f, header=True)