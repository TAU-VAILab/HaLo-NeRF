
import os
import pickle
from matplotlib import pyplot as plt
import torch
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



path = '/storage/chendudai/data/clipseg_ft_crops_refined_plur_newcrops_10epochs/badshahi_mosque/horizontal/clipseg_ft/minarets/'
l_dir = os.listdir(path)
colormap = plt.get_cmap('jet')

html_out = open(os.path.join(path, "clipseg_ft_horiz.html"), "w")
print('<head><meta charset="UTF-8"></head>', file=html_out)
print("<h1>Results</h1>", file=html_out)
x = torch.zeros(1)
for l in l_dir:
    if l == '0029.pickle':
        continue
    print(l)
    try:
        with open(os.path.join(path,l), 'rb') as f:
            semantics_gt =pickle.load(f)
        print('s')
    except:
        with open(os.path.join(path,l), 'rb') as f:
            semantics_gt = torch.Tensor(torch.load(f))
        print('e')

    fig = plt.figure()
    fig, axis = plt.subplots(1,2, figsize=(20,4))
    axis[0].imshow(semantics_gt)
    plt.savefig(path + '1_.png')
    print_img(path + '1_.png',html_out)
    # os.remove(path + '1_.png')

print("<hr>", file=html_out)
html_out.close()





