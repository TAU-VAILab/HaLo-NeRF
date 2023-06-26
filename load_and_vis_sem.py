import imageio
import torch
from PIL import Image
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('module://backend_interagg')
# from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import cv2
import imageio
from PIL import Image


# path = '/storage/chendudai/repos/Ha-NeRF/sem_results/notre_dame_ours_clipsegFtCrops_maxIter12500_top50And50_getOccRetHorizAndClipFtRetHoriz/results/phototourism/for_metric/top_50_nEpochs10/towers_ds2/900_semantic.png'
# path_img = '/storage/chendudai/data/notre_dame_front_facade/dense/images/0900.jpg'

num = '056'
path = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/save/results/phototourism/milano_portals_ours/'
path_gt_image = '/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/dense/images/' + '0056'+ '.jpg'

# path = '/storage/chendudai/repos/Ha-NeRF/save/results/phototourism/0209_domes_180523/'
# path_gt_image = '/storage/chendudai/data/0209_megaDepth_mosque/dense/images/' + '0'+ num+ '.jpg'


path_pred = os.path.join(path,f'{num}_semantic.png')
path_img = os.path.join(path,f'{num}.png')

pred_original = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

img = cv2.imread(path_img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

gt_img = cv2.imread(path_gt_image)
gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
gt_img = cv2.resize(gt_img, dsize=(gt_img.shape[1]//2, gt_img.shape[0]//2), interpolation=cv2.INTER_CUBIC)

# First method for vis
pred = np.expand_dims(pred_original, axis=2)
overlay_yellow = np.concatenate([pred,pred,np.zeros_like(pred)], axis=2)    # yellow
overlay_green = np.concatenate([np.zeros_like(pred),pred,np.zeros_like(pred)], axis=2)   # green
# alpha = 0.5
# result_green = cv2.addWeighted(img, 1 - alpha,overlay_green, alpha, 0)
# result_yellow = cv2.addWeighted(img, 1 - alpha,overlay_yellow, alpha, 0)
# result_green_with_rgb = cv2.addWeighted(gt_img, 1 - alpha,overlay_green, alpha, 0)
# result_yellow_with_rgb = cv2.addWeighted(gt_img, 1 - alpha,overlay_yellow, alpha, 0)

pred_original = np.expand_dims(pred_original/255, axis=2)
result_green = pred_original * overlay_green + (1-pred_original) * img
result_green = result_green.astype('uint8')
result_yellow = pred_original * overlay_yellow + (1-pred_original) * img
result_yellow = result_yellow.astype('uint8')

result_green_with_rgb = pred_original * overlay_green + (1-pred_original) * gt_img
result_green_with_rgb = result_green_with_rgb.astype('uint8')
result_yellow_with_rgb = pred_original * overlay_yellow + (1-pred_original) * gt_img
result_yellow_with_rgb = result_yellow_with_rgb.astype('uint8')


imageio.imwrite(os.path.join(path, f'{num}_new_vis_green.png'), result_green)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow.png'), result_yellow)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_green_with_rgb_gt.png'), result_green_with_rgb)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow_with_rgb_gt.png'), result_yellow_with_rgb)
imageio.imwrite(os.path.join(path, f'{num}_rgb_gt.png'), gt_img)



img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.expand_dims(img_gray, axis=2)
img_gray = np.concatenate([img_gray,img_gray,img_gray],axis=2)

gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
gt_img_gray = np.expand_dims(gt_img_gray, axis=2)
gt_img_gray = np.concatenate([gt_img_gray,gt_img_gray,gt_img_gray],axis=2)


# First method for vis
# result_green = cv2.addWeighted(img_gray, 1 - alpha,overlay_green, alpha, 0)
# result_yellow = cv2.addWeighted(img_gray, 1 - alpha,overlay_yellow, alpha, 0)
# result_green_with_rgb = cv2.addWeighted(gt_img_gray, 1 - alpha,overlay_green, alpha, 0)
# result_yellow_with_rgb = cv2.addWeighted(gt_img_gray, 1 - alpha,overlay_yellow, alpha, 0)
result_green = pred_original * overlay_green + (1-pred_original) * img_gray
result_green = result_green.astype('uint8')
result_yellow = pred_original * overlay_yellow + (1-pred_original) * img_gray
result_yellow = result_yellow.astype('uint8')

result_green_with_rgb = pred_original * overlay_green + (1-pred_original) * gt_img_gray
result_green_with_rgb = result_green_with_rgb.astype('uint8')
result_yellow_with_rgb = pred_original * overlay_yellow + (1-pred_original) * gt_img_gray
result_yellow_with_rgb = result_yellow_with_rgb.astype('uint8')


imageio.imwrite(os.path.join(path, f'{num}_new_vis_green_gray.png'), result_green)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow_gray.png'), result_yellow)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_green_with_rgb_gt_gray.png'), result_green_with_rgb)
imageio.imwrite(os.path.join(path, f'{num}_new_vis_yellow_with_rgb_gt_gray.png'), result_yellow_with_rgb)
imageio.imwrite(os.path.join(path, f'{num}_rgb_gt_gray.png'), gt_img_gray)


print('saved in:')
print(path)
