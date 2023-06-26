import os

path = r'C:\Users\ASUS\Documents\TAU\Thesis\Towers_Of_Babel\data\WikiScenes3D\WikiScenes3D'
dir = os.listdir(path)

for d in dir:
    x = os.path.join(path,d)
    if not os.path.isdir(x):
        continue
    x_2 = os.listdir(x)
    for d_2 in x_2:
        y = os.path.join(path, d, d_2)
        if not os.path.isdir(y):
            continue
        y_2 = os.listdir(y)
        for z in y_2:
            if z == 'images.txt':
                file_path = os.path.join(path, d, d_2,z)
                file_stat = os.stat(file_path)
                kb = file_stat.st_size / 1024
                if kb > 10000:
                    print(file_path)