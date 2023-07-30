import numpy as np
import os
import cv2
import imageio


num = '056'
path = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/save/results/phototourism/milano_portals_ours/'
path_gt_image = '/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/dense/images/' + '0056'+ '.jpg'


path_pred = os.path.join(path,f'{num}_semantic.png')
path_img = os.path.join(path,f'{num}.png')

pred_original = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

img = cv2.imread(path_img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

gt_img = cv2.imread(path_gt_image)
gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
gt_img = cv2.resize(gt_img, dsize=(gt_img.shape[1]//2, gt_img.shape[0]//2), interpolation=cv2.INTER_CUBIC)

pred = np.expand_dims(pred_original, axis=2)
overlay_yellow = np.concatenate([pred,pred,np.zeros_like(pred)], axis=2)    # yellow
overlay_green = np.concatenate([np.zeros_like(pred),pred,np.zeros_like(pred)], axis=2)   # green

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
