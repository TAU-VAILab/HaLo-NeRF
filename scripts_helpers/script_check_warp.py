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


cat = ['portals','domes', 'windows'] #'portals', 'domes', 'minarets', 'windows'

for c in cat:

    path_warp = '/storage/chendudai/for_morris/warps_fixed_mosques_and_hurba/hurba/' + c
    path_images = '/storage/chendudai/data/hurba/dense/images/'
    path2save = '/storage/chendudai/for_morris/gt_warps/vis/hurba_fixed/' + c

    warps= os.listdir(path_warp)
    warps.sort()

    imgs = os.listdir(path_images)
    imgs.sort()
    os.makedirs(path2save, exist_ok=True)
    i = 0
    html_out = open(os.path.join(path2save, "vis.html"), "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h1>Results</h1>", file=html_out)

    print(c)
    for w in warps:
        name = w.split('.')[0].split('-')[-1]
        for img in imgs:
            if img.split('.')[0] == name:
                i += 1
                print(f'img: {img}, i: {i}')
                I = Image.open(os.path.join(path_images, img))
                W = Image.open(os.path.join(path_warp, w))
                I = np.array(I)
                W = np.array(W)
                W = cv2.cvtColor(W, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(I, 0.5, W, 0.5, 0)
                f = plt.figure()
                plt.imshow(result)
                plt.savefig(os.path.join(path2save, img))
                print(f"<br><b>{img}</b><br>", file=html_out)
                print_img(os.path.join(path2save, img), html_out)
                os.remove(os.path.join(path2save, img))


    print("<hr>", file=html_out)
    html_out.close()