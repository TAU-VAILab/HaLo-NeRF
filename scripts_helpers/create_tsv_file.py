import csv
import glob


root_dir = 'data/scene_name/dense/images/'
save_dir = 'data/scene_name/WikiScenes.tsv'

with open(save_dir, 'wt', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['filename', 'id', 'split', 'dataset'])
    id = 0
    for filename in glob.iglob(root_dir + '**/**', recursive=True):
        if filename.endswith(".jpg"):

            if id % 10 == 0:
                split = 'test'
            else:
                split = 'train'
            try:
                filename = filename.split('/')[-1]
                tsv_writer.writerow([filename, str(id), split, 'WikiScenes'])
                id += 1

            except:
                print(filename)
