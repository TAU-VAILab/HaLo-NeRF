import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import base64

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


cat = ['portals', 'windows', 'domes'] #'portals', 'domes', 'minarets', 'windows', 'towers', 'spires'

for c in cat:

    # path_warp = '/storage/chendudai/for_morris/warps_fixed_mosques_and_hurba/hurba/' + c
    # path_images = '/storage/chendudai/data/hurba/dense/images/'
    # path2save = '/storage/chendudai/for_morris/gt_warps/vis/hurba_fixed/' + c

    path_warp = '/storage/chendudai/data/gt_warps_with_manually_100/hurba/' + c
    path_images = '/storage/chendudai/data/hurba/dense/images/'
    path2save = '/storage/chendudai/data/vis_gt_warps_with_manually_100_comparison_retrieval_jet/hurba/' + c
    path_metrics ='/storage/chendudai/repos/HaLo-NeRF/sem_results/ours/hurba/results/phototourism/for_metric/top_150_nEpochs10/' + c + '_ds2'
    path_metrics_2 ='/storage/chendudai/repos/HaLo-NeRF/sem_results/ours_without_retrieval/hurba/results/phototourism/for_metric/top_4000_nEpochs10/' + c + '_ds2'

    warps= os.listdir(path_warp)
    warps.sort()

    imgs = os.listdir(path_images)
    imgs.sort()
    os.makedirs(path2save, exist_ok=True)
    i = 0
    html_out = open(os.path.join(path2save, "vis.html"), "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h1>Results</h1>", file=html_out)

    colormap = plt.get_cmap('jet')

    print(c)
    for w in warps:
        name = w.split('.')[0].split('_')[0]
        for img in imgs:
            if img.split('.')[0] == name:
                i += 1
                print(f'img: {img}, i: {i}')
                I = Image.open(os.path.join(path_images, img))
                W = Image.open(os.path.join(path_warp, w))
                M = Image.open(os.path.join(path_metrics, str(int(name)).zfill(3) +'_semantic.png'))
                txt_file = open(os.path.join(path_metrics, str(int(name)).zfill(3) +'_semantic_metric.txt'), "r")
                txt_file.readline()
                score = float(txt_file.readline()[26:32])

                M_2 = Image.open(os.path.join(path_metrics_2, str(int(name)).zfill(3) + '_semantic.png'))
                txt_file_2 = open(os.path.join(path_metrics_2, str(int(name)).zfill(3) + '_semantic_metric.txt'), "r")
                txt_file_2.readline()
                try:
                    score_2 = float(txt_file_2.readline()[26:32])
                except:
                    score_2 = np.nan
                    print('error')


                I = np.array(I)
                W = np.array(W)

                # M = np.array(M)
                # M_2 = np.array(M_2)
                M = 255 - np.array(M)
                M_2 = 255 - np.array(M_2)


                try:
                    W = cv2.cvtColor(W, cv2.COLOR_GRAY2BGR)
                except:
                    pass
                try:
                    # M = cv2.cvtColor(M, cv2.COLOR_GRAY2BGR)
                    M = colormap(M)
                except:
                    pass
                try:
                    # M_2 = cv2.cvtColor(M_2, cv2.COLOR_GRAY2BGR)
                    M_2 = colormap(M_2)
                except:
                    pass
                GT = cv2.addWeighted(I, 0.5, W, 0.5, 0)

                # res = cv2.addWeighted(I, 0.5, M, 0.5, 0)
                # res_2 = cv2.addWeighted(I, 0.5, M_2, 0.5, 0)
                res = 0.5*(M[:,:,:3] + GT/255)
                res_2 = 0.5*(M_2[:,:,:3] + GT/255)

                fig, axis = plt.subplots(1, 3, figsize=(20,4))
                axis[0].imshow(GT)
                axis[0].title.set_text('GT')
                im = axis[1].imshow(res, cmap=colormap)
                axis[1].title.set_text(f'result - ours, score: {score}')
                plt.colorbar(im, ax=axis[1])
                im2 = axis[2].imshow(res_2, cmap=colormap)
                axis[2].title.set_text(f'result - without retrieval, score: {score_2}')
                plt.colorbar(im2, ax=axis[2])

                plt.savefig(os.path.join(path2save, img))
                print(f"<br><b>{img}</b><br>", file=html_out)

                print_img(os.path.join(path2save, img), html_out)
                os.remove(os.path.join(path2save, img))


    print("<hr>", file=html_out)
    html_out.close()