import os, shutil
from pathlib import Path
import fnmatch
import json

# input_path = '/storage/chendudai/data/WikiScenes/cathedrals'
# output_path = '/storage/chendudai/data/Categories'
input_path = r'C:\Users\chend\Documents\TAU\Thesis\Towers_Of_Babel\data\WikiScenes1200px\cathedrals'
output_path = r'C:\Users\chend\Documents\TAU\Thesis\data\Categories'
captions = []
path_images = []
list_categories = []
cnt = 0
classes = ["portal", "window", "tower"]

for root, dirnames, filenames in os.walk(Path(input_path)):
    for filename in fnmatch.filter(filenames, '*.json'):
        path_json = os.path.join(root, filename)

        with open(path_json, 'r', encoding='utf-8') as f:
            my_data = json.load(f)
        for image in my_data['pictures']:
            caption = my_data['pictures'][image]['caption']
            file_path = os.path.join(path_json[:-13], 'pictures', image)
            captions.append(caption)

            # Extract Categories For The Images
            categories = []
            path_categories = Path(file_path)
            while path_categories.name != 'cathedrals':
                son_folder = path_categories.name
                path_categories = path_categories.parent
                if path_categories.name == 'pictures':
                    path_categories = path_categories.parent
                    continue

                for filename_category in fnmatch.filter(filenames, '*.json'):
                    path_json_category = os.path.join(path_categories, filename_category)
                    with open(path_json_category, 'r', encoding='utf-8') as f:
                        folder_data = json.load(f)

                    category = list(folder_data['pairs'].keys())[int(son_folder)]
                    categories.append(category.lower())

            for c in classes:
                if c in categories[0]:
                    if 'exterior' in categories[-2]:
                        path_images.append(file_path)
                        list_categories.append(categories[0])



print("len[path_images] = ", len(path_images))


# Create Dir
for c in classes:
    folder = os.path.join(output_path, c)
    os.makedirs(folder, exist_ok=True)

# Copy files
for i, file in enumerate(path_images):
    for c in classes:
        if c in list_categories[i].lower():
            filename = Path(file).name
            try:
                shutil.copy2(file, os.path.join(output_path, c, filename))
            except:
                print(filename)