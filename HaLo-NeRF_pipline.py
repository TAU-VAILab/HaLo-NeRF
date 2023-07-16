import train_mask_grid_sample
import config.halonerf_config as Cfg_file
from config.opt import get_opts
import utils.save_semantic_for_metric
import utils.calculate_metrics
import utils.visualize
import numpy as np
import os

## get Opts
opts = Cfg_file.get_opts()
prompts = opts.prompts.split(';')

metrics = []
ts_list = []

hparam_train = get_opts()

for prompt in prompts:
    print(f'Start: {prompt}')
    cfg = Cfg_file.getCfg(opts, prompt)

    ts_list = []
    if os.path.exists(os.path.join(cfg['path_gt'], cfg['category'])):
        list_dir = os.listdir(os.path.join(cfg['path_gt'], cfg['category']))
        ts_list = [int(f[:4]) for f in list_dir if f.endswith('jpg')]

    if ts_list == []:
        print('no category')
        # raise ValueError('no category')


    ## Train The Semantic Ha-NeRF
    if cfg['train_HaloNeRF_flag']:
        print('Train HaLo-NeRF')
        hparam_train.root_dir = cfg['root_dir']
        hparam_train.save_dir = cfg['save_dir']
        hparam_train.img_downscale = cfg['img_downscale']
        hparam_train.num_epochs = cfg['num_epochs']
        hparam_train.batch_size = cfg['batch_size']
        hparam_train.lr = cfg['lr']
        hparam_train.N_vocab = cfg['N_vocab']
        hparam_train.encode_a = cfg['encode_a']
        hparam_train.encode_random = cfg['encode_random']
        hparam_train.continue_train_semantic = cfg['continue_train_semantic']
        hparam_train.enable_semantic = cfg['enable_semantic']
        hparam_train.semantics_dir = cfg['semantics_dir']
        hparam_train.exp_name = cfg['exp_name']
        hparam_train.ckpt_path = cfg['ckpt_path']
        hparam_train.files_to_run = cfg['files_to_run']
        hparam_train.neg_files = cfg['neg_files']
        hparam_train.use_semantic_function = cfg['use_semantic_function']
        hparam_train.max_steps = cfg['max_steps']

        hparam_train.threshold = cfg['threshold']

        train_mask_grid_sample.main_train_mask_grid_sample(hparam_train)


    # in case it didn't end all the epochs
    if not os.path.exists(cfg['ckpt_path_eval']):
        parent = cfg['ckpt_path_eval'].split('/')[:-1]
        parent = '/'.join(parent)
        l_dir = os.listdir(parent)
        l_dir = [f.split('.')[0].split('=')[-1] for f in l_dir]
        max_epoch = max(l_dir)
        real_ep = cfg['ckpt_path_eval'].split('epoch=')[1].split('.ckpt')[0]
        old_path = cfg['ckpt_path_eval']
        cfg['ckpt_path_eval'] = cfg['ckpt_path_eval'].replace(real_ep, max_epoch)
        cfg['ckpt_path_eval'] = os.path.join(parent, 'epoch='+ cfg['ckpt_path_eval'].split('epoch=')[1])
        new_path = cfg['ckpt_path_eval']
        print(f'ckpt: {old_path} does not exist, takes instead: {new_path}')

    ## save_semantic_for_metric
    if cfg['save_for_metric_flag']:
        print('save semantic for metric')
        utils.save_semantic_for_metric.main_eval(ts_list, cfg['root_dir'], cfg['N_vocab'], cfg['scene_name'], cfg['ckpt_path_eval'], cfg['save_dir_eval'], cfg['top_k_files'], cfg['num_epochs'])



    ## Calc Metrics
    if cfg['calc_metrics_flag']:
        print('calc metrics')
        metrics.append(utils.calculate_metrics.main_metrics(cfg['path_pred'], cfg['path_gt'], cfg['category'], cfg['top_k_files'], cfg['num_epochs'], opts.scene_name, cfg['category'], ts_list))


    ## Visualizaiton
    if cfg['vis_flag']:
        print('vis')
        utils.visualize.main_vis(cfg['save_training_vis'], cfg['files'], ts_list, cfg['root_dir'], cfg['N_vocab'], cfg['scene_name'], cfg['ckpt_path_eval'], cfg['save_dir_eval'], cfg['folder2save'],  cfg['path_gt'], prompt, cfg['top_k_files'], cfg['num_epochs'])


## Calc Metrics - avarage_per_class_metrics

if cfg['calc_metrics_flag']:

    ap_mean = np.mean([x['ap'] for x in metrics])
    ba_mean = np.mean([x['ba'] for x in metrics])
    jaccard_mean = np.mean([x['jaccard'] for x in metrics])
    dice_mean = np.mean([x['dice'] for x in metrics])

    print(ap_mean)
    print(ba_mean)
    print(jaccard_mean)
    print(dice_mean)

    k = cfg['top_k_files']
    num_epochs = cfg['num_epochs']

    PRED_THRESHOLD = cfg['PRED_THRESHOLD']
    with open(cfg['path_pred'] +'/avarage_per_class_metrics.txt', 'w') as f:
        f.write("Macro-averaged metrics (per-class average):\n")
        f.write('\tAP (average precision):\t' + str(ap_mean) + '\n')
        f.write('\tBalanced accuracy:\t' + str(ap_mean) + '\n')
        f.write('\tJaccard score (IoU):\t' + str(jaccard_mean) + f' (Threshold: {PRED_THRESHOLD})\n')
        f.write('\tDice score (F1):\t' + str(dice_mean) + f' (Threshold: {PRED_THRESHOLD})\n')
        f.write('\n')


print('Done')