import numpy as np
import torch
import os

# os.system('! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip')
# os.system('! unzip -d weights -j weights.zip')
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import imageio
import pickle

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict=False)


# load and normalize image
path = '/storage/chendudai/data/0_1_undistorted/dense/images'
categories_file = '/storage/chendudai/repos/Ha-NeRF/categories.txt'
list_images = os.listdir(path)
folder = '0_1/'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((352, 352)),
])

with open(categories_file) as f:
    lines = f.readlines()

for img_name in list_images:
    name_int = int(img_name.split('.')[0])
    print(img_name)
    name = img_name.split('.')[0]
    cat = lines[name_int].split('\t')[1].split('\n')[0]
    img = Image.open(os.path.join(path, img_name))
    img = img.resize((352, 352))
    img_tensor = transform(img)
    img_tensor_2 = transform_2(img)

    prompts = []
    threshold = 0.5
    threshold_str = str(threshold)
    if 'window' in cat:
        prompts = ['a window', 'a photo of a window', 'a photo of windows',
                   'a photo of a window of the cathedral',
                   'a photo of windows of the cathedral', 'The windows of the cathedral']
    elif 'statue' in cat:
        prompts = ['statue', 'a photo of a statue', 'a statue of the cathedral', 'the statues of the cathedral',
                   'a photo of a statue of the cathedral', 'a photo of the statues of the cathedral']
    elif 'door' in cat:
        prompts = ['door', 'a photo of a door', 'a door of the cathedral', 'the doors of the cathedral',
                   'a photo of a door of the cathedral', 'a photo of the doors of the cathedral']

    elif 'tower' in cat:
        prompts = ['tower', 'a photo of a tower', 'a tower of the cathedral', 'the tower of the cathedral',
                   'a photo of a tower of the cathedral', 'a photo of the tower of the cathedral']
    elif 'top part' in cat:
        prompts = ['top part', 'a photo of a top part', 'a top part of the cathedral', 'the top part of the cathedral',
                   'a photo of a top part of the cathedral', 'a photo of the top part of the cathedral']
    elif 'spire' in cat:
        prompts = ['spire', 'a photo of a spire', 'a spire of the cathedral', 'the spires of the cathedral',
                   'a photo of a spire of the cathedral', 'a photo of the spires of the cathedral']
    elif 'flank' in cat:
        prompts = ['flank', 'a photo of a flank', 'a flank of the cathedral', 'the flanks of the cathedral',
                   'a photo of a flank of the cathedral', 'a photo of the flanks of the cathedral']
    elif 'apse' in cat:
        prompts = ['apse', 'apse exterior', 'a photo of apse', 'apse of the cathedral',
                   'a photo of a apse of the cathedral', 'facade']

    elif 'pinnacles' in cat:
        prompts = ['pinnacles', 'pinnacle', 'a photo of pinnacles', 'pinnacles of the cathedral',
                   'a photo of a pinnacles of the cathedral', 'facade']

    elif 'madonnina' in cat:
        prompts = ['madonnina', 'a photo of madonnina', 'madonnina of the cathedral',
                   'a photo of a madonnina of the cathedral', 'statue', 'facade']

    else: #'facade' in cat:
        prompts = ['facade', 'a photo of a facade', 'a facade of the cathedral', 'the facade of the cathedral',
                   'a photo of a facade of the cathedral', 'a photo of the facade of the cathedral']

    prompts.append(cat)

    # predict
    with torch.no_grad():
        preds = model(img_tensor.repeat(len(prompts),1,1,1), prompts)[0]

    # show prediction
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(img)
    [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
    [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    plt.tight_layout()
    os.makedirs('./results/' + folder, exist_ok=True)
    plt.savefig('./results/' + folder + 'prediction' + name + '.png')
    # plt.show()

     # show prediction with threshold overlay
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(img)
    input_image = img.resize((352, 352))
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # img_tensor = transform(input_image) / 255
    for i in range(len(prompts)):
        x = torch.sigmoid(preds[i][0])
        x[x<threshold] = 0
        x[x>=threshold] = 1
        y = 0.2*img_tensor[0,:,:] + 0.8*x
        img_tensor[0, :, :] = img_tensor_2[0, :, :] * (1 - x)
        img_tensor[1, :, :] = img_tensor_2[1, :, :] * (1 - x)
        img_tensor[2, :, :] = img_tensor_2[2, :, :] * (1 - x) + (x)
        ax[i+1].imshow(img_tensor.permute(1,2,0))
        ax[i+1].text(0, -15, prompts[i])
    plt.tight_layout()
    plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()






    threshold = 0.25
    threshold_str = str(threshold)

    # show prediction with threshold overlay
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(img)
    input_image = img.resize((352, 352))
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # img_tensor = transform(input_image) / 255
    for i in range(len(prompts)):
        x = torch.sigmoid(preds[i][0])
        x[x<threshold] = 0
        x[x>=threshold] = 1
        y = 0.2*img_tensor[0,:,:] + 0.8*x
        img_tensor[0, :, :] = img_tensor[0, :, :] * (1 - x)
        img_tensor[1, :, :] = img_tensor[1, :, :] * (1 - x)
        img_tensor[2, :, :] = img_tensor[2, :, :] * (1 - x) + (255 * x)
        ax[i+1].imshow(img_tensor.permute(1,2,0))
        ax[i+1].text(0, -15, prompts[i])
    plt.tight_layout()
    plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()
