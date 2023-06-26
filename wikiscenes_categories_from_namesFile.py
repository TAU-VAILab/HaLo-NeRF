import os, shutil
from pathlib import Path
import fnmatch
import json

data_path = r'C:\Users\ASUS\Documents\TAU\Thesis\Towers_Of_Babel\data\WikiScenes1200px\cathedrals'
input_path = './names_98_3.txt'
captions = []
path_images = []
list_categories = []
cnt = 0

f = open(input_path, "r", encoding='utf-8')
lines = f.read()
x = lines.split('\n')
x = x[:-1]
categories = []
names = []
for string in x:
    path_images.append(string.split('\t')[1])
    names.append(string.split('\t')[0])


for j, i in enumerate(path_images):
    print(j)
    p = Path(i)
    try:
        parent = p.parent.parent.parent
        son = p.parent.parent
        son = str(son).split('\\')[-1]
        files = os.listdir(os.path.join(data_path, parent))
    except:
        p = Path(i[1:])
        parent = p.parent.parent.parent
        son = p.parent.parent
        son = str(son).split('\\')[-1]
        files = os.listdir(os.path.join(data_path, parent))

    for f in files:
        if f[-5:] == '.json':
            with open(os.path.join(data_path, parent,f), 'r', encoding='utf-8') as f:
                json_file = json.load(f)

            category = list(json_file['pairs'].keys())[int(son)]
            categories.append(category.lower())


with open("output.txt", "w", encoding="utf-8") as f:
    for i, c in enumerate(categories):
        f.write(names[i])
        f.write('\t')
        f.write(c)
        f.write('\n')