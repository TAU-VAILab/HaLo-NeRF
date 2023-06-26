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

model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
model.eval()
model.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict=False)

# load and normalize image

path = '/storage/chendudai/data/0_1_undistorted/dense/images'
list_images = os.listdir(path)
input_image = '/storage/chendudai/data/0_1_undistorted/dense/images/0779.jpg'
folder = '0779/'
# names = ['_window', '_statue', '_door', '_facade', '_tower', '_top_part', '_spire']
# names = ['_door']
names = ['_window']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

for name in names:
    if name == '_window':
        # prompts = ['a window', 'a photo of a window', 'a photo of windows',
        #            'a photo of a window of the cathedral',
        #            'a photo of windows of the cathedral', 'The windows of the cathedral']
        prompts = ['a photo of windows']
    if name == '_statue':
        prompts = ['statue', 'a photo of a statue', 'a statue of the cathedral', 'the statues of the cathedral',
                   'a photo of a statue of the cathedral', 'a photo of the statues of the cathedral']
    if name == '_door':
        # prompts = ['door', 'a photo of a door', 'a door of the cathedral', 'the doors of the cathedral',
        #            'a photo of a door of the cathedral', 'a photo of the doors of the cathedral']
        prompts = ['a photo of a door']

    if name == '_facade':
        # prompts = ['facade', 'a photo of a facade', 'a facade of the cathedral', 'the facade of the cathedral',
        #            'a photo of a facade of the cathedral', 'a photo of the facade of the cathedral']
        prompts = ['a photo of a facade']
    if name == '_tower':
        prompts = ['tower', 'a photo of a tower', 'a tower of the cathedral', 'the tower of the cathedral',
                   'a photo of a tower of the cathedral', 'a photo of the tower of the cathedral']
    if name == '_top_part':
        prompts = ['top part', 'a photo of a top part', 'a top part of the cathedral', 'the top part of the cathedral',
                   'a photo of a top part of the cathedral', 'a photo of the top part of the cathedral']
    if name == '_spire':
        prompts = ['spire', 'a photo of a spire', 'a spire of the cathedral', 'the spires of the cathedral',
                   'a photo of a spire of the cathedral', 'a photo of the spires of the cathedral']
    if name == '_flank':
        # prompts = ['flank', 'a photo of a flank', 'a flank of the cathedral', 'the flanks of the cathedral',
        #            'a photo of a flank of the cathedral', 'a photo of the flanks of the cathedral']
        prompts = ['the flanks of the cathedral']
    if name == '_pinnacles':
        prompts = ['pinnacles']

    # for input_image in list_images:
    img = Image.open(os.path.join(path, input_image))
    img = transform(img).unsqueeze(0)


    threshold = 0.5
    threshold_str = str(threshold)

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

    # # show prediction
    # _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
    # [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    # plt.tight_layout()
    # os.makedirs('./results/' + folder, exist_ok=True)
    # plt.savefig('./results/' + folder + 'prediction' + name + '.png')
    # plt.show()


    if name == '_window':
        for i, prompt in enumerate(prompts):
            # with open('./results/' + folder + prompt.replace(" ", "_") + '.pkl', 'wb') as handle:
            # input_image.split('.')[0]
            y = '0779'
            os.makedirs('./results/window/' + y, exist_ok=True)
            with open('./results/window/' + y + '.pickle', 'wb') as handle:
                mask = torch.sigmoid(preds[i][0])
                # mask[mask<0.5] = 0
                # mask[mask>=0.5] = 1
                x = './results/window/' + y + '.png'
                x1 = './results/window/' + y + '_1.png'
                imageio.imwrite(x, mask)
                imageio.imwrite(x1,  (10*mask+ img).squeeze(dim=0).permute(1,2,0))
                torch.save(mask, handle)
    #
    # # show prediction with threshold overlay
    # _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # input_image = input_image.resize((352, 352))
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # img_tensor = transform(input_image) / 255
    # for i in range(len(prompts)):
    #     x = torch.sigmoid(preds[i][0])
    #     x[x<threshold] = 0
    #     x[x>=threshold] = 1
    #     y = 0.2*img_tensor[0,:,:] + 0.8*x
    #     img_tensor[0, :, :] = img_tensor[0, :, :] * (1 - x)
    #     img_tensor[1, :, :] = img_tensor[1, :, :] * (1 - x)
    #     img_tensor[2, :, :] = img_tensor[2, :, :] * (1 - x) + (255 * x)
    #     ax[i+1].imshow(img_tensor.permute(1,2,0))
    #     ax[i+1].text(0, -15, prompts[i])
    # plt.tight_layout()
    # plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()
    #
    #
    #
    #
    #
    #
    # threshold = 0.25
    # threshold_str = str(threshold)
    #
    # # show prediction with threshold overlay
    # _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # input_image = input_image.resize((352, 352))
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # img_tensor = transform(input_image) / 255
    # for i in range(len(prompts)):
    #     x = torch.sigmoid(preds[i][0])
    #     x[x<threshold] = 0
    #     x[x>=threshold] = 1
    #     y = 0.2*img_tensor[0,:,:] + 0.8*x
    #     img_tensor[0, :, :] = img_tensor[0, :, :] * (1 - x)
    #     img_tensor[1, :, :] = img_tensor[1, :, :] * (1 - x)
    #     img_tensor[2, :, :] = img_tensor[2, :, :] * (1 - x) + (255 * x)
    #     ax[i+1].imshow(img_tensor.permute(1,2,0))
    #     ax[i+1].text(0, -15, prompts[i])
    # plt.tight_layout()
    # plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()
