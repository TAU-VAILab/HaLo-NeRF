import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt


def green_func(number_path, model, proc, flag_use_mask):
    num = number_path.split('/')[-1].zfill(3)
    num_4 = number_path.split('/')[-1].zfill(4)
    path_pred = os.path.join(number_path, num+'_semantic.png')
    path_gt_image = os.path.join(number_path, num+'_rgb_gt.png')
    path_gt_image_2 = os.path.join(number_path, num+'_rgb_gt.jpg')

    # path_gt_image = '/storage/chendudai/data/notre_dame_front_facade/dense/images/' + num_4 + '.jpg'
    # path_gt_image_2 = '/storage/chendudai/data/notre_dame_front_facade/dense/images/' + num_4 + '.JPG'
    #
    path_nerf_rgb = os.path.join(number_path, num+'.png')
    # path2save = os.path.join(number_path.split('data_figures/')[0], 'save_figures_blur500_withAlpha_test' ,number_path.split('data_figures/')[1])
    path2save = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/res_clipseg/' + number_path.split('clipseg/')[1]
    os.makedirs(path2save, exist_ok=True)

    try:
        pred_original = 255-cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
    except:
        return

    # maxi = pred_original.max()
    # mini = pred_original.min()
    # pred_original = (pred_original - mini) / (maxi - mini)
    # pred_original = (pred_original * 255).astype('uint8')

    w, h = pred_original.shape
    m = min(w,h)
    r = 500/m

    pred_original = cv2.resize(pred_original, (int(pred_original.shape[1] * r), int(pred_original.shape[0] * r)))

    pred_original_blur = cv2.GaussianBlur(pred_original, (13,13), 10)

    # pred_original_blur = np.array(pred_original_blur/255 > 0.5).astype('uint8') * 255


    pred_original_opp = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

    # maxi = pred_original_opp.max()
    # mini = pred_original_opp.min()
    # pred_original_opp = (pred_original_opp - mini) / (maxi - mini)
    # pred_original_opp = (pred_original_opp * 255).astype('uint8')
    #
    w, h = pred_original_opp.shape
    m = min(w,h)
    r = 500/m

    pred_original_opp = cv2.resize(pred_original_opp, (int(pred_original_opp.shape[1] * r), int(pred_original_opp.shape[0] * r)))

    pred_original_opp_blur = cv2.GaussianBlur(pred_original_opp, (11,11), 10)

    # pred_original_opp_blur = np.array(pred_original_opp_blur/255 > 0.5).astype('uint8') * 255


    if os.path.isfile(path_nerf_rgb):
        pred_rgb = Image.open(path_nerf_rgb)
        pred_rgb = pred_rgb.resize((pred_original.shape[1], pred_original.shape[0]))

    if os.path.isfile(path_gt_image):
        gt_img = cv2.imread(path_gt_image)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
    elif os.path.isfile(path_gt_image_2):
        gt_img = cv2.imread(path_gt_image_2)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)

    gt_img = cv2.resize(gt_img, dsize=(pred_original.shape[1], pred_original.shape[0]), interpolation=cv2.INTER_CUBIC)


    C = np.zeros((256, 4), dtype=np.float32)
    C[:, 1] = 1.
    # C[:, 0] = 1. #yellow

    # C[:, 0] = 1. #RED


    C[:, -1] = np.linspace(0, 1, 256)
    cmap_ = ListedColormap(C)

    if flag_use_mask:
        with torch.no_grad():
            inp = proc(images=gt_img, return_tensors='pt')
            out = model(**inp)
        L = out.logits[0]

        S = L.softmax(dim=0)[2].numpy()
        S = cv2.resize(S, (gt_img.shape[1], gt_img.shape[0]))
    else:
        S = 0

    # plt.imshow(pred_rgb)
    # plt.imshow(pred_original, cmap=cmap_, alpha=np.asarray(pred_original) / 255)
    # plt.axis('off')
    # plt.savefig('/storage/chendudai/data/x.png', bbox_inches="tight", pad_inches=0)
    # plt.close()

    pred_masked = pred_original * (1 - S)


    plt.imshow(gt_img)
    plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original) / 255)
    # plt.imshow(pred_masked, cmap=cmap_)
    plt.axis('off')
    plt.savefig(os.path.join(path2save, num +'_with_gt.png'), bbox_inches="tight", pad_inches=0)
    plt.close()

    pred_masked = pred_original_opp * (1 - S)

    plt.imshow(gt_img)
    plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_opp) / 255)
    # plt.imshow(pred_masked, cmap=cmap_)

    plt.axis('off')
    plt.savefig(os.path.join(path2save, num +'_with_gt_opp.png'), bbox_inches="tight", pad_inches=0)
    plt.close()

    if os.path.isfile(path_nerf_rgb):
        pred_masked = pred_original * (1 - S)

        plt.imshow(pred_rgb)
        plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original) / 255)
        # plt.imshow(pred_masked, cmap=cmap_)

        plt.axis('off')
        plt.savefig(os.path.join(path2save, num +'_with_pred.png'), bbox_inches="tight", pad_inches=0)
        plt.close()

        pred_masked = pred_original_opp * (1 - S)

        plt.imshow(pred_rgb)
        plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_opp) / 255)
        # plt.imshow(pred_masked, cmap=cmap_)

        plt.axis('off')
        plt.savefig(os.path.join(path2save, num +'_with_pred_opp.png'), bbox_inches="tight", pad_inches=0)
        plt.close()

    pred_masked = pred_original_blur * (1 - S)


    plt.imshow(gt_img)
    plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_blur) / 255)
    # plt.imshow(pred_masked, cmap=cmap_)

    plt.axis('off')
    plt.savefig(os.path.join(path2save, num + '_with_gt_blur.png'), bbox_inches="tight", pad_inches=0)
    plt.close()

    pred_masked = pred_original_opp_blur * (1 - S)


    plt.imshow(gt_img)
    plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_opp_blur) / 255)
    # plt.imshow(pred_masked, cmap=cmap_)

    plt.axis('off')
    plt.savefig(os.path.join(path2save, num + '_with_gt_opp_blur.png'), bbox_inches="tight", pad_inches=0)
    plt.close()

    if os.path.isfile(path_nerf_rgb):
        pred_masked = pred_original_blur * (1 - S)


        plt.imshow(pred_rgb)
        plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_blur) / 255)
        # plt.imshow(pred_masked, cmap=cmap_)

        plt.axis('off')
        plt.savefig(os.path.join(path2save, num + '_with_pred_blur.png'), bbox_inches="tight", pad_inches=0)
        plt.close()

        pred_masked = pred_original_opp_blur * (1 - S)

        plt.imshow(pred_rgb)
        plt.imshow(pred_masked, cmap=cmap_, alpha=np.asarray(pred_original_opp_blur) / 255)
        # plt.imshow(pred_masked, cmap=cmap_)

        plt.axis('off')
        plt.savefig(os.path.join(path2save, num + '_with_pred_opp_blur.png'), bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f'saved in: {path2save}')

proc = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")

flag_use_mask = False
# base_directory = '/storage/chendudai/data/data_figures/'
base_directory = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/clipseg/'

for method in os.listdir(base_directory):
    method_path = os.path.join(base_directory, method)
    # method_path = '/storage/chendudai/data/data_figures/clipseg_ft'
    for scene_name in os.listdir(method_path):
        scene_path = os.path.join(method_path, scene_name)
        # scene_path = '/storage/chendudai/data/data_figures/ours/st_paul'
        if os.path.isdir(scene_path):
            # Traverse through object directories
            for object_name in os.listdir(scene_path):
                object_path = os.path.join(scene_path, object_name)
                # object_path = '/storage/chendudai/data/data_figures/ours/buckingham_palace/lamps'
                # object_path = '/storage/chendudai/data/data_figures/ours/st_paul/towers'
                if os.path.isdir(object_path):
                    # Traverse through number directories
                    for number_folder in os.listdir(object_path):
                        number_path = os.path.join(object_path, number_folder)
                        # number_path = '/storage/chendudai/data/data_figures/lerf/notre_dame/towers/900'
                        if os.path.isdir(number_path):

                            num = number_path.split('/')[-1].zfill(3)
                            path_pred = os.path.join(number_path, num + '_semantic.png')
                            if os.path.isfile(path_pred):
                                # Generate the "Y" function and save it to the save directory
                                green_func(number_path, model, proc, flag_use_mask)



