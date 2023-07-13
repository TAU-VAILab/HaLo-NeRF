import numpy as np
import pandas
import argparse
import pickle
import os
def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--semantics_dir', type=str, default='/home/cc/students/csguests/chendudai/Thesis/data/clipseg_ft_crops_refined_plur_newcrops_10epochs/milano/horizontal/clipseg_ft')
    parser.add_argument('--scene_name', type=str, default='')

    # Flags
    parser.add_argument('--clipseg_flag', default=False, action="store_true") # Flase as dafault
    parser.add_argument('--use_refined_clipseg', type=bool, default=True)
    parser.add_argument('--is_indoor_scene', default=False, action="store_true")

    parser.add_argument('--train_HaloNeRF_flag', default=False, action="store_true")
    parser.add_argument('--save_for_metric_flag', default=False, action="store_true")
    parser.add_argument('--calc_metrics_flag', default=False, action="store_true")
    parser.add_argument('--vis_flag', default=False, action="store_true")
    parser.add_argument('--save_training_vis', default=False, action="store_true")

    parser.add_argument('--use_vis_prompt', default=False, action="store_true")
    parser.add_argument('--use_threshold', default=False, action="store_true")
    parser.add_argument('--neg_prec', type=float, default=0)

    # main
    parser.add_argument('--root_dir', type=str, default='/storage/chendudai/data/st_pauls_cathedral/',   #'/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted'    #/home/cc/students/csguests/chendudai/Thesis/data/0209_megaDepth_mosque/
                        help='root directory of dataset')
    parser.add_argument('--prompts', type=str, default="towers;windows;portals") #spires;windows;portals;facade   towers;windows;portals   #statue

    # Clipseg
    parser.add_argument('--top_k_files', type=int, default=150)

    parser.add_argument('--xls_path', type=str, default='/storage/chendudai/data/ft_clip_sims_v0.3-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24-nodups.csv')   #'/home/cc/students/csguests/chendudai/Thesis/data/ft_clip_sims_v0.3-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24-nodups.csv' #retrieval_clip_outdoor_020523.csv
    parser.add_argument('--save_dir', type=str, default='./sem_results/0_1_undistorted_ft_clip')

    parser.add_argument('--vis_prompt_path', type=str, default='/storage/chendudai/data/visual_prompts_top100_v3-clipseg-rd64.pk')  # '/home/cc/students/csguests/chendudai/Thesis/data/visual_prompts_top100_v3-clipseg-rd64.pk'

    parser.add_argument('--use_rgb_loss', type=bool, default=False)

    # HaLo-NeRF Training
    parser.add_argument('--exp_name', type=str, default='top50_ds2_epoch2_lr5e-5')
    parser.add_argument('--img_downscale', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)

    parser.add_argument('--use_semantic_function', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, default=7225)
    parser.add_argument('--N_vocab', type=int, default=4000)
    parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/st_pauls_cathedral/epoch=19.ckpt') #'/storage/chendudai/repos/Ha-NeRF/sem_results/0_1_undistorted_ft_clip_top15_ds2_ep2_lr5-5_continueTrainRGB_facadeOnly/ckpts/top745_ds2_epoch2_lr5e-5/facade/epoch=0.ckpt'
    # parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/0_1_withoutSemantics/epoch=15.ckpt')
    # parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/0209_megaDepth_mosque/epoch=13.ckpt')
    # parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/0237_megaDepth_mosque/epoch=1.ckpt')
    # parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/notre_dame_front_facade/epoch=1.ckpt')

    parser.add_argument('--encode_a', type=bool, default=True)
    parser.add_argument('--encode_random', type=bool, default=True)
    parser.add_argument('--continue_train_semantic', type=bool, default=True)
    parser.add_argument('--enable_semantic', type=bool, default=True)
    parser.add_argument('--max_steps', type=int, default=12500)

    # save_semantic_for_metric


    # Calc Metrics
    parser.add_argument('--path_gt', type=str, default='/storage/chendudai/data/manually_gt_masks_st_paul/')    #'/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_0_1/'
    parser.add_argument('--PRED_THRESHOLD', type=float, default=0.5)

    # Vis
    return parser.parse_args()


def get_k_files(k, csv_path, prompt, neg_prec):
    xls_file = pandas.read_csv(csv_path)
    col = xls_file[prompt]

    col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    col_sorted_ascending = col.sort_values(by=prompt, ascending=True)

    files_pos = col_sorted_descending[:k]
    files_neg = col_sorted_ascending[:int(k*neg_prec)]

    names_pos = xls_file['filename'][files_pos.index]
    names_neg = xls_file['filename'][files_neg.index]

    pos_confidence_values = files_pos.values.tolist()

    return [names_pos.values.tolist(), names_neg.values.tolist(), pos_confidence_values]


def get_k_files_clip(k, csv_path, prompt, scene_name):
    xls_file = pandas.read_csv(csv_path)
    xls_file = xls_file[xls_file['building'] == scene_name]
    col = xls_file[prompt]

    col_sorted_descending = col.sort_values(by=prompt, ascending=False)

    files_pos = col_sorted_descending[:k]

    names_pos = xls_file['base_fn'][files_pos.index]

    pos_confidence_values = files_pos.values.tolist()
    neg = []
    return [names_pos.values.tolist(), neg, pos_confidence_values]

def get_files_with_threshold(csv_path, prompt, neg_prec, threshold=0.24):
    # Note: variable "prompt" is a List[str] with one element
    assert type(prompt) is list and len(prompt) == 1, 'Wrong format'
    prompt_string = prompt[0]
    xls_file = pandas.read_csv(csv_path)
    col = xls_file[prompt_string]

    col_above_thresh = col[col > threshold]
    n_chosen = len(col_above_thresh)
    assert n_chosen > 0, 'No matching images!'



    # col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    col_sorted_ascending = col.sort_values(ascending=True)
    # files_descending = col_sorted_descending > 0.24
    files_ascending = col_sorted_ascending[:int(n_chosen*neg_prec)]

    # names_descending = xls_file['filename'][files_descending.index]
    names_positive = xls_file['filename'][col_above_thresh.index]
    names_negative = xls_file['filename'][files_ascending.index]

    pos_confidence_values = files_ascending.values.tolist()

    return [names_positive.values.tolist(), names_negative.values.tolist(), pos_confidence_values]

def getCfg(opts, prompt):
    cfg = {}

    # Flags
    cfg['clipseg_flag'] = opts.clipseg_flag

    cfg['train_HaloNeRF_flag'] = opts.train_HaloNeRF_flag
    cfg['save_for_metric_flag'] = opts.save_for_metric_flag
    cfg['calc_metrics_flag'] = opts.calc_metrics_flag
    cfg['vis_flag'] = opts.vis_flag

    cfg['root_dir'] = opts.root_dir

    cfg['max_steps'] = opts.max_steps
    cfg['threshold'] = opts.threshold

    cfg['use_rgb_loss'] = opts.use_rgb_loss

    ## Clipseg
    cfg['top_k_files'] = opts.top_k_files
    cfg['prompt'] = prompt
    original_prompt = prompt
    cfg['xls_path'] = opts.xls_path

    cfg['prompt'] = [prompt]


    cfg['path_images'] = cfg['root_dir'] + '/dense/images'
    cfg['folder2save'] = opts.save_dir.split('/')[-1] + '/clipseg/' + cfg['prompt'][0].replace(' ','_').replace("\'",'') + '/top_' + str(cfg['top_k_files']) + '/'
    cfg['neg_prec'] = opts.neg_prec
    cfg['threshold'] = opts.threshold

    cfg['use_vis_prompt'] = opts.use_vis_prompt
    cfg['vis_prompt_data'] = []
    if cfg['use_vis_prompt']:
        with open(opts.vis_prompt_path, 'rb') as file:
            cfg['vis_prompt_data'] = pickle.load(file)

    if opts.use_threshold:
        [cfg['files'], cfg['neg_files'], cfg['pos_confidence_values']]= get_files_with_threshold(cfg['xls_path'], cfg['prompt'], cfg['neg_prec'], cfg['threshold'])
    else:
        if not opts.is_indoor_scene:
            [cfg['files'], cfg['neg_files'], cfg['pos_confidence_values']] = get_k_files(cfg['top_k_files'], cfg['xls_path'], ["a picture of a cathedral's facade"], cfg['neg_prec'])
            print('geo occ (facade) retrieval: ')
            print(cfg['files'])
        else:
            cfg['files'] = os.listdir(cfg['path_images'])
            print(cfg['files'])
            cfg['neg_files'], cfg['pos_confidence_values'] = [], []


    ## HaLo-NeRF Training
    cfg['save_dir'] = opts.save_dir
    cfg['img_downscale'] = opts.img_downscale
    cfg['num_epochs'] = opts.num_epochs
    cfg['batch_size'] = opts.batch_size
    cfg['lr'] = opts.lr
    cfg['N_vocab'] = opts.N_vocab
    cfg['exp_name'] = opts.exp_name + '/' + prompt
    cfg['ckpt_path'] = opts.ckpt_path  # For continue training the semantics
    cfg['encode_a'] = opts.encode_a
    cfg['encode_random'] = opts.encode_random
    cfg['continue_train_semantic'] = opts.continue_train_semantic
    cfg['enable_semantic'] = opts.enable_semantic
    cfg['use_semantic_function'] = opts.use_semantic_function
    cfg['use_refined_clipseg'] = opts.use_refined_clipseg

    cfg['files_to_run'] = cfg['files']

    ## save_semantic_for_metric

    if 'spire' in prompt:
        cfg['category'] = 'spires'
    elif 'portal' in prompt or 'door' in prompt:
        cfg['category'] = 'portals'
    elif 'facade' in prompt:
        cfg['category'] = 'facade'
    elif 'window' in prompt:
        cfg['category'] = 'windows'
    elif 'tower' in prompt:
        cfg['category'] = 'towers'
    elif 'dome' in prompt:
        cfg['category'] = 'domes'
    elif 'minaret' in prompt:
        cfg['category'] = 'minarets'
    else:
        cfg['category'] = prompt
        print('No Category')
        # raise ValueError('no category')

    if opts.use_refined_clipseg:
        cfg['semantics_dir'] = os.path.join(opts.semantics_dir, cfg['category'])

        if not os.path.exists(cfg['semantics_dir']):
            raise ValueError('no semantics_dir')

    else:
        cfg['semantics_dir'] = '/'.join(opts.save_dir.split('/')[:-1]) + '/' + cfg['folder2save']  # load clipseg results for training the semantic

    cfg['scene_name'] = prompt.replace(' ','_').replace("\'",'') + '_ds' + str(cfg['img_downscale']) # Where to save
    cfg['ckpt_path_eval'] = cfg['save_dir'] + '/ckpts/' + cfg['exp_name'] + '/epoch=' + str(cfg['num_epochs']-1) + '.ckpt'
    cfg['save_dir_eval'] = cfg['save_dir']

    ## Calc Metrics
    cfg['PRED_THRESHOLD'] = opts.PRED_THRESHOLD

    cfg['path_gt'] = opts.path_gt

    k = cfg['top_k_files']
    nEpochs = cfg['num_epochs']
    cfg['path_pred'] = cfg['save_dir'] + '/results/phototourism/for_metric/' + f'top_{k}_nEpochs{nEpochs}/' + cfg['scene_name']

    ## Visuzalize
    cfg['save_training_vis'] = opts.save_training_vis

    return cfg



