import os
import pickle
import numpy as np

cats = ['altar']
for cat in cats:
    path = '/home/cc/students/csguests/chendudai/Thesistopא/data/morris_npy/seville_indoor/' + cat + '/'
    # path = '/storage/chendudai/data/morris_npy/seville_indoor/' + cat + '/'
    dir_list = os.listdir(path)

    for i in dir_list:
        print(i)
        # p = path + d +'/'
        # x = os.listdir(p)

        # for i in x:
        #     j = i.replace(d + '_','')
        #     os.rename(p + i,p + j)

        # for i in d:
        j = i.replace(cat + '_','')
        # j = i.replace(cat + '-','')
        j = j.replace('.npy','.pickle')
        try:
            with open(path + i, 'rb') as f:
                z = np.load(f)
        except:
            continue
        os.rename(path + i, path + j)
        with open(path + j, 'wb') as f:
            pickle.dump(z, f)
