import os
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, f1_score
import json


gt_folder = '/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_0_1/'

GT_THRESHOLD = 0.5
PRED_THRESHOLD = 0.5

class MetricCalculator:

    def __init__(self):
        pass

    def process_image(self, gt_array, pred_array):

        gt_mask = gt_array > GT_THRESHOLD
        gt_mask_flat = gt_mask.ravel()
        pred_array_flat = pred_array.ravel()

        pred_array_flat_thresh = pred_array_flat > PRED_THRESHOLD

        # calculate metrics
        AP = average_precision_score(gt_mask_flat, pred_array_flat)

        tpr = (gt_mask_flat * pred_array_flat).sum() / gt_mask_flat.sum()
        tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / (1 - gt_mask_flat).sum()

        balanced_acc = (tpr + tnr) / 2

        jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

        dice = f1_score(gt_mask_flat, pred_array_flat_thresh)

        return [AP, balanced_acc, jscore, dice]
        # print('AP (average precision):\t', AP)
        # print('Balanced accuracy:\t', balanced_acc)
        # print('Jaccard score (IoU):\t', jscore, f'(Threshold: {self.PRED_THRESHOLD})')
        # print('Dice score (F1):\t', dice, f'(Threshold: {self.PRED_THRESHOLD})')





calc = MetricCalculator()


promots = os.listdir(gt_folder)
all_metrics = []
all_labels = []

for p in promots:
    top_k_folders = os.listdir(os.path.join(gt_folder,p))
    clipseg_folder = os.path.join(os.path.join(gt_folder, p, 'clipseg_results'))
    for k in top_k_folders:
        gt_fn = os.path.join(gt_folder, p, k)
        if not k.endswith('.jpg') and not k.endswith('.JPG') :
            continue
        gt_img = Image.open(gt_fn).convert('L')
        gt_img = 1 - np.asarray(gt_img) / 255
        name = gt_fn.split('/')[-1].split('_')[0]
        pred_fn = os.path.join(clipseg_folder, name + '.pickle')
        clipseg_res = torch.load(pred_fn)
        clipseg_res = torch.nn.functional.interpolate(clipseg_res.unsqueeze(dim=0).unsqueeze(dim=0),
                                                   size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze().numpy()

        metrics = calc.process_image(gt_img, clipseg_res)

        all_metrics += [metrics]
        all_labels += [p]


        with open(os.path.join(clipseg_folder + '_' + name + '_metric.txt'), 'w') as file:
            file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
            file.write(f'\tAP (average precision):\t {metrics[0]}\n')
            file.write(f'\tBalanced accuracy:\ {metrics[1]}\n')
            file.write(f'\tJaccard score (IoU):\t {metrics[2]}, Threshold: {PRED_THRESHOLD}\n')
            file.write(f'\tDice score (F1):\t {metrics[3]}, Threshold: {PRED_THRESHOLD}')
            file.close()

per_label_metrics = {p: [] for p in promots}
for p, m in zip(all_labels, all_metrics):
    per_label_metrics[p].append(m)
per_label_avgs = {
    p: [
        np.mean([x[i] for x in vals])
        for i in range(4)
    ]
    for p, vals in per_label_metrics.items()
}
# macro_averages = [
#     np.mean([avgs[i] for p, avgs in per_label_avgs.items()])
#     for i in range(4)
# ]

print('Per label averages:')
print(per_label_avgs)
out_fn = os.path.join(gt_folder, 'per_label_metrics.json')
with open(out_fn, 'w') as file:
    json.dump(per_label_avgs, file)
print('Saved to', out_fn)

# with open(os.path.join(gt_folder + 'all_metric.txt'), 'w') as file:
#     file.write(f'per class (GT: {gt_fn}; preds: {pred_fn})\n')
#     file.write(f'\tAP (average precision):\t {macro_averages[0]}\n')
#     file.write(f'\tBalanced accuracy:\ {metrics[1]}\n')
#     file.write(f'\tJaccard score (IoU):\t {macro_averages[2]}, Threshold: {PRED_THRESHOLD}\n')
#     file.write(f'\tDice score (F1):\t {macro_averages[3]}, Threshold: {PRED_THRESHOLD}')
#     file.close()