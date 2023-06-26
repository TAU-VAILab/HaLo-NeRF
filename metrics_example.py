from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, jaccard_score, f1_score

class MetricCalculator:
    
    GT_THRESHOLD = 0.5
    PRED_THRESHOLD = 0.5
    
    def __init__(self):
        pass

    def process_image(self, pred_fn, gt_fn):
        pred_img = Image.open(pred_fn).convert('L')
        gt_img = Image.open(gt_fn).convert('L')

        pred_array = 1 - np.asarray(pred_img) / 255
        gt_array = 1 - np.asarray(gt_img) / 255
        # 1 - x : assumes background is x=1

        gt_uniq = np.unique(gt_array)
        gt_nuniq = len(gt_uniq)
        if gt_nuniq != 2:
            print(f'Warning: Ground truth mask contains {gt_nuniq} values unique:', *gt_uniq)
            print('Using threshold', self.GT_THRESHOLD)

        gt_mask = gt_array > self.GT_THRESHOLD

        gt_mask_flat = gt_mask.ravel()
        pred_array_flat = pred_array.ravel()

        pred_array_flat_thresh = pred_array_flat > self.PRED_THRESHOLD

        # calculate metrics
        AP = average_precision_score(gt_mask_flat, pred_array_flat)

        tpr = (gt_mask_flat * pred_array_flat).sum() / gt_mask_flat.sum()
        tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / (1 - gt_mask_flat).sum()

        balanced_acc = (tpr + tnr) / 2

        jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

        dice = f1_score(gt_mask_flat, pred_array_flat_thresh)

        print('AP (average precision):\t', AP)
        print('Balanced accuracy:\t', balanced_acc)
        print('Jaccard score (IoU):\t', jscore, f'(Threshold: {self.PRED_THRESHOLD})')
        print('Dice score (F1):\t', dice, f'(Threshold: {self.PRED_THRESHOLD})')


if __name__ == "__main__":
    calculator = MetricCalculator()
    calculator.process_image('23_pred.png', '0023_mask.JPG')