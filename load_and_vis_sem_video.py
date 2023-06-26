import numpy as np
import os
import cv2
import imageio
from matplotlib import pyplot as plt


def unique(list1):
    unique_list = []
    for x in list1:
        try:
            y = int(x)
        except:
            continue
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


path = '/storage/chendudai/repos/Ha-NeRF/save/results/phototourism/milano_window_ours/'
l_dir = os.listdir(path)
l_dir = [f[:3]for f in l_dir]
l_dir = unique(l_dir)
l_dir = sorted(l_dir)

results_green = []
results_yellow = []
results_green_gray = []
results_yellow_gray = []

for num in l_dir:

    path_pred = os.path.join(path,f'{num}_semantic.png')
    path_img = os.path.join(path,f'{num}.png')

    pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    # First method for vis
    pred = np.expand_dims(pred, axis=2)
    overlay_yellow = np.concatenate([pred,pred,np.zeros_like(pred)], axis=2)    # yellow
    overlay_green = np.concatenate([np.zeros_like(pred),pred,np.zeros_like(pred)], axis=2)   # green

    pred = pred / 255
    result_green = pred * overlay_green + (1 - pred) * img
    result_green = result_green.astype('uint8')
    result_yellow = pred * overlay_yellow + (1 - pred) * img
    result_yellow = result_yellow.astype('uint8')

    results_green += [result_green]
    results_yellow += [result_yellow]

    # imageio.imwrite(os.path.join(path, f'{num}_new_vis_green.png'), results_green)
    # imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow.png'), results_yellow)
    #


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.expand_dims(img_gray, axis=2)
    img_gray = np.concatenate([img_gray, img_gray, img_gray], axis=2)

    result_green_gray = pred * overlay_green + (1-pred) * img_gray
    result_green_gray = result_green_gray.astype('uint8')
    result_yellow_gray = pred * overlay_yellow + (1-pred) * img_gray
    result_yellow_gray = result_yellow_gray.astype('uint8')

    results_green_gray += [result_green_gray]
    results_yellow_gray += [result_yellow_gray]

    # imageio.imwrite(os.path.join(path, f'{num}_new_vis_green_gray.png'), result_green_gray)
    # imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow_gray.png'), result_yellow_gray)

res_iterp = []

for a in np.arange(0,1.01,0.05):
    iterp = a * img + (1-a) * result_green
    iterp = iterp.astype('uint8')
    res_iterp.append(iterp)

res_iterp.extend([img]*15)
results_green.extend(res_iterp)
results_green_rev = results_green[::-2]
results_green_with_rev = results_green + results_green_rev

res_iterp = []

for a in np.arange(0,1.01,0.05):
    iterp = a * img + (1-a) * result_yellow
    iterp = iterp.astype('uint8')
    res_iterp.append(iterp)

res_iterp.extend([img]*15)
results_yellow.extend(res_iterp)
results_yellow_rev = results_yellow[::-2]
results_yellow_with_rev = results_yellow + results_yellow_rev


imageio.mimsave(os.path.join(path, 'vid_sem_green_with_rev.mp4'), results_green_with_rev, fps=24)
imageio.mimsave(os.path.join(path, 'vid_sem_yellow_with_rev.mp4'), results_yellow_with_rev, fps=24)



# imageio.mimsave(os.path.join(path, 'vid_sem_green.mp4'), results_green, fps=24)
# imageio.mimsave(os.path.join(path, 'vid_sem_yellow.mp4'), results_yellow, fps=24)
#
# imageio.mimsave(os.path.join(path, 'vid_sem_green_gray.mp4'), results_green_gray, fps=24)
# imageio.mimsave(os.path.join(path, 'vid_sem_yellow_gray.mp4'), results_yellow_gray, fps=24)

