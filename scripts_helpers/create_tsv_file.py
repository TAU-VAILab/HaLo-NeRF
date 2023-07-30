import csv
import glob


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
