import csv
import os
import glob

# images = os.listdir(r'C:\Users\chen\Documents\TAU\Thesis\Ha-NeRF\data\WikiScenes\Wells_Cathedral\dense\images')
# root_dir = r'C:\Users\chen\Documents\TAU\Thesis\Towers_Of_Babel\data\WikiScenes1200px\cathedrals\50'

# root_dir = '/storage/chendudai/data/st_pauls_cathedral/dense/images/'
# save_dir = '/storage/chendudai/data/st_pauls_cathedral/WikiScenes.tsv'

root_dir = '/home/cc/students/csguests/chendudai/Thesis/data/0148_megaDepth/dense/images/'
save_dir = '/home/cc/students/csguests/chendudai/Thesis/data/0148_megaDepth/WikiScenes.tsv'

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
                # tsv_writer.writerow([filename[84:].replace('\\', '/'), str(id), split, 'florence_cathedral'])
                id += 1

            except:
                print(filename)
