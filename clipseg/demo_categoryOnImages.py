import numpy as np
import torch
import os

import glob
import base64
# os.system('! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip')
# os.system('! unzip -d weights -j weights.zip')
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import matplotlib
import PIL
import torchvision.transforms as T
import cv2

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


def main_clipseg(files, path_images, folder2save, prompts, vis_prompt=[], pos_confidence_values=0):

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict=False)


    list_images = os.listdir(path_images)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])


    os.makedirs('./sem_results/' + folder2save, exist_ok=True)


    # save HTML
    html_out = open(os.path.join('./sem_results/' + folder2save, "clipseg.html"), "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h1>Results</h1>", file=html_out)

    for i, img_name in enumerate(files):
            if img_name[-4:] != '.png' and img_name[-4:] != '.jpg' and img_name[-4:] != '.JPG':
                continue
            print(img_name)
            name = img_name.split('.')[0]
            try:
                img = Image.open(os.path.join(path_images, img_name))
            except:
                try:
                    img_name = img_name.replace('.jpg','.JPG')
                    img = Image.open(os.path.join(path_images, img_name))
                except:
                    try:
                        img_name = img_name.replace('.JPG', '.png')
                        img = Image.open(os.path.join(path_images, img_name))
                    except:
                        print(f'no image: {img_name}')
                        continue
                # try:
                #     img = Image.open(os.path.join(path_images, img_name))
                # except:
                #     print('no file')
                #     imgs = os.listdir(path_images)
                #     idx = min(int(img_name.split('.')[0]), len(imgs)-1)
                #     img_name = imgs[idx]
                #     img = Image.open(os.path.join(path_images, img_name))



            img = img.resize((352, 352))
            img_tensor = transform(img)

            if vis_prompt != []:
                with torch.no_grad():
                    word = prompts[0].split()[-1]
                    preds = model(img_tensor.repeat(len(prompts),1,1,1), vis_prompt[word].unsqueeze(dim=0).cpu())[0]

            else:
                # predict
                with torch.no_grad():
                    preds = model(img_tensor.repeat(len(prompts),1,1,1), prompts)[0]


            mask = torch.sigmoid(preds[0][0])
            # mask_before = mask

            # mask = mask / mask.max()
            # mask[mask < 0.5] = 0
            # mask[mask >= 0.5] = 1

            # transform_to_image = T.ToPILImage()
            # mask_img = transform_to_image(np.round((1-mask).unsqueeze(dim=0)*255))
            # th_val, th_img = cv2.threshold(np.uint8(mask_img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # hist = torch.histc(mask,255)
            # mask = th_img / 255
            # mask = torch.Tensor(mask)

            colormap = plt.get_cmap('jet')



            if __name__ == '__main__':
                with open(folder2save + name + '.pickle', 'wb') as handle:
                    torch.save(mask, handle)

                plt.imsave(folder2save + name + '_pred_clipseg.png',mask, cmap=colormap)
                img.save(folder2save + name + '.png')
                continue

            else:
                with open('./sem_results/' + folder2save + name + '.pickle', 'wb') as handle:
                    torch.save(mask, handle)

                plt.imsave('./sem_results/' + folder2save + name + '_pred_clipseg.png', mask, cmap=colormap)
                img.save('./sem_results/' + folder2save + name + '.png')

            fig, axis = plt.subplots(1, 3, figsize=(20, 4))
            # fig.suptitle(f'prompt: {prompts[0]}, Clip retrival confidence: {pos_confidence_values[i][0]}, Retrival Order: {i+1}')
            fig.suptitle(f'prompt: {prompts[0]}, Clip retrival confidence: {pos_confidence_values[i]}, Retrival Order: {i+1}')
            axis[0].imshow(img)
            axis[0].title.set_text('rgb gt')
            im = axis[1].imshow(mask, cmap=colormap)
            axis[1].title.set_text('clipseg pred')
            axis[2].imshow(img)
            axis[2].imshow(mask, cmap=colormap, alpha=0.5)
            axis[2].title.set_text(f'clipseg pred blend')

            # axis[3].plot(hist)
            # x = np.zeros(len(hist))
            # x[np.int(th_val)] = 10000
            # axis[3].plot(x)
            # axis[3].title.set_text(f'pred histogram and Ostu value')
            # im2 = axis[4].imshow(mask_before, cmap=colormap)
            # axis[4].title.set_text(f'mask before')
            # axis[5].imshow(img)
            # axis[5].imshow(mask_before, cmap=colormap, alpha=0.5)
            # axis[5].title.set_text(f'mask before blend')

            for ax in axis:
                ax.axis('off')
            plt.tight_layout()
            fig.colorbar(im)
            path2save = os.path.join('./sem_results/' + folder2save + name + '_clipseg.png')
            fig.savefig(path2save)



            print(f"<br><b>{os.path.basename(path2save)}</b><br>", file=html_out)
            print_img(path2save, html_out)

    print("<hr>", file=html_out)
    html_out.close()



if __name__ == '__main__':
    # files = ['0025.jpg', '0634.JPG', '0637.jpg','0645.JPG', '0764.jpg'] # facade
    cat = 'minarets'

    files = os.listdir('/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_badshahi/' + cat +'/')
    # prompts = "a picture of a cathedral's " + cat
    prompts = cat
    files = [f[:-9] + f[-4:] for f in files]
    path_images = '/home/cc/students/csguests/chendudai/Thesis/data/badshahi_mosque/dense/images/'
    folder2save = '/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_badshahi/' + cat + '/clipseg_results/'
    os.makedirs(folder2save, exist_ok=True)
    main_clipseg(False, files, path_images, folder2save, prompts)