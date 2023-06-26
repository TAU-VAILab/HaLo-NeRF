import os
from matplotlib import pyplot as plt
import torch
import pickle
from PIL import Image
import base64
import pandas as pd

from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

def print_img(image_path, output_file):
    """
    Encodes an image into html.
    image_path (str): Path to image file
    output_file (file): Output html page
    """
    if os.path.exists(image_path):
        img = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        print(
            '<img src="data:image/png;base64,{0}">'.format(img),
            file=output_file,
        )


geometric_occlusions = True

if geometric_occlusions:
    csv_path = '/home/cc/students/csguests/chendudai/Thesis/data/milano_geometric_occlusions.csv'
else:
    csv_path = '/home/cc/students/csguests/chendudai/Thesis/data/retrieval_clip_outdoor_020523-ft.csv'

scene_name = 'milano'

categories = ['portals/', 'windows/', 'domes/', 'minarets/']
# categories = ['portals/', 'windows/', 'spires/']

for cat in categories:
    prompt = cat.split('/')[0]
    path = '/home/cc/students/csguests/chendudai/Thesis/data/clipseg_ft_crops_10epochs/milano/windowed/'    #
    threshold = 0.2

    if geometric_occlusions:
        save_dir = path + f'vis_clipseg_ft_crops_10epochs_geoOccRetrivelSorted_thresh{threshold}/' + cat
    else:
        save_dir = path + f'vis_clipseg_ft_crops_10epochs_clipFtRetrieval_thresh{threshold}/' + cat

    path_images = '/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/dense/images/'


    k = 50
    path_cat = path + cat

    if geometric_occlusions:
        dataFrame = pd.read_csv(csv_path)
        dataFrame = dataFrame.sort_values(by="a picture of a cathedral's facade", ascending=False)
        list_name = dataFrame[:]
        list_name = list_name['filename'][list_name.index]
        list_names = list_name.values.tolist()

    else:
        xls_file = pd.read_csv(csv_path)
        xls_file = xls_file[xls_file['building'] == scene_name]
        col = xls_file[prompt]
        col_sorted_descending = col.sort_values(ascending=False)
        files_pos = col_sorted_descending[:k]
        names_pos = xls_file['base_fn'][files_pos.index]
        list_names = names_pos.values.tolist()




    dir_list_cat = os.listdir(path_cat)
    dir_list_images = os.listdir(path_images)

    os.makedirs(save_dir, exist_ok=True)
    colormap = plt.get_cmap('jet')

    html_out = open(os.path.join(save_dir, "vis_results.html"), "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h1>Results</h1>", file=html_out)


    j = 0


    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])


    for i in list_names:
        print(j)
        j = j + 1

        with open(os.path.join(path_images, i), 'rb') as f:
            img = Image.open(f).convert('RGB')


        path = os.path.join(path_cat, i.replace(i.split('.')[1],'pickle'))
        try:
            with open(path, 'rb') as f:
                # semantics_gt = torch.Tensor(pickle.load(f))
                semantics_gt = torch.Tensor(torch.load(f))
        except:
            continue

        img_w, img_h = img.size
        # img = img.resize((img_w//2, img_h//2), Image.LANCZOS)

        semantics_gt = torch.nn.functional.interpolate(semantics_gt.unsqueeze(dim=0).unsqueeze(dim=0), size=(img_h, img_w))
        semantics_gt = semantics_gt.squeeze(dim=0).permute(1, 2, 0)



        # fig, axis = plt.subplots(1, 5, figsize=(20, 4))
        # axis[0].imshow(img)
        # axis[0].title.set_text('rgb gt')
        # im = axis[1].imshow(semantics_gt, cmap=colormap)
        # axis[1].title.set_text('clipseg pred')
        # axis[2].imshow(img)
        # axis[2].imshow(semantics_gt, cmap=colormap, alpha=0.5)
        # axis[2].title.set_text(f'clipseg pred blend')
        #
        # semantics_gt_thresh = semantics_gt
        # semantics_gt_thresh[semantics_gt_thresh<threshold] = 0
        # semantics_gt_thresh[semantics_gt_thresh>=threshold] = 1
        # im2 = axis[3].imshow(semantics_gt, cmap=colormap)
        # axis[3].title.set_text('clipseg pred with threshold')
        # axis[4].imshow(img)
        # axis[4].imshow(semantics_gt, cmap=colormap, alpha=0.5)
        # axis[4].title.set_text(f'clipseg pred blend with threshold')

        fig, axis = plt.subplots(1, 6, figsize=(20, 4))
        axis[0].imshow(img)
        axis[0].title.set_text('rgb gt')
        im = axis[1].imshow(semantics_gt, cmap=colormap)
        axis[1].title.set_text('finetuned clipseg pred')
        axis[2].imshow(img)
        axis[2].imshow(semantics_gt, cmap=colormap, alpha=0.5)
        axis[2].title.set_text(f'finetuned clipseg pred blend')

        semantics_gt_thresh = semantics_gt >= threshold

        # semantics_gt = 1 / (1 + torch.exp(-15 * (semantics_gt - 0.2)))

        # semantics_gt_thresh[semantics_gt_thresh<threshold] = 0
        # semantics_gt_thresh[semantics_gt_thresh>=threshold] = 1

        axis[3].imshow(img)
        im1 = axis[3].imshow(semantics_gt_thresh, cmap=colormap, alpha=0.5)
        axis[3].title.set_text(f'finetuned clipseg pred blend with threshold')
        # axis[3].title.set_text(f'finetuned clipseg pred blend with sigmoid')

        img_resized = img.resize((352, 352))
        img_tensor = transform(img_resized)
        with torch.no_grad():
            preds = model(img_tensor.repeat(len(prompt),1,1,1), prompt)[0]
        mask_clipseg = torch.sigmoid(preds[0][0])
        axis[4].imshow(img_resized)
        axis[4].imshow(mask_clipseg, cmap=colormap, alpha=0.5)
        axis[4].title.set_text(f'regular clipseg pred blend')

        mask_clipseg_thresh = mask_clipseg

        # mask_clipseg_thresh = 1 / (1 + torch.exp(-15 * (mask_clipseg_thresh - 0.2)))

        mask_clipseg_thresh[mask_clipseg_thresh < threshold] = 0
        mask_clipseg_thresh[mask_clipseg_thresh >= threshold] = 1

        axis[5].imshow(img_resized)
        axis[5].imshow(mask_clipseg_thresh, cmap=colormap, alpha=0.5)
        axis[5].title.set_text(f'regular clipseg pred blend with threshold')
        # axis[5].title.set_text(f'regular clipseg pred blend with sigmoid')

        #
        # threshold_2 = semantics_gt.max() / 3.3
        # semantics_gt_thresh = semantics_gt >= threshold_2
        # # semantics_gt_thresh[semantics_gt_thresh < threshold] = 0
        # # semantics_gt_thresh[semantics_gt_thresh >= threshold] = 1
        #
        # axis[4].imshow(img)
        # axis[4].imshow(semantics_gt_thresh, cmap=colormap, alpha=0.5)
        # axis[4].title.set_text(f'blend with adaptive threshold, max pred: {semantics_gt.max()}')
        #

        for ax in axis:
            ax.axis('off')
        plt.tight_layout()
        fig.colorbar(im)
        full_path_save = save_dir + i.replace('.pickle','.jpg')
        fig.savefig(full_path_save)

        print(f"<br><b>{os.path.basename(full_path_save)}</b><br>", file=html_out)
        print(f"<br><b>{str(j)}</b><br>", file=html_out)
        print_img(full_path_save, html_out)

        if j >= k:
            break

    print("<hr>", file=html_out)
    html_out.close()
