import pandas
from argparse import ArgumentParser
import os

def get_opts():
    parser = ArgumentParser()


    # Flags
    parser.add_argument('--train_HaloNeRF_flag', default=False, action="store_true", help='a flag for training the 3D localization head')
    parser.add_argument('--save_for_metric_flag', default=False, action="store_true", help='a flag for saving the images predictions for metric calculations')
    parser.add_argument('--calc_metrics_flag', default=False, action="store_true", help='a flag for calculate the metrics using the images predictions and the ground truth')
    parser.add_argument('--ignore_csv_use_all_images', default=False, action="store_true", help='a flag for ignoring the csv retrieval file, and train the 3D localization on all the images')

    # main
    parser.add_argument('--root_dir', type=str, default='data/st_pauls_cathedral/',
                        help='root directory of dataset')
    parser.add_argument('--prompts', type=str, default="towers;windows;portals", help='text prompts to train the localization head, seprated with ;')

    parser.add_argument('--csv_retrieval_path', type=str, default='data/ft_clip_sims_v0.3-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24-nodups.csv', help='the path for the csv file to retrieve the images for the HaLo-NeRF training')
    parser.add_argument('--top_k_files', type=int, default=150, help='the number of images to retrieve from the csv retrieval file')

    # HaLo-NeRF Training
    parser.add_argument('--exp_name', type=str, default='top50_ds2_epoch2_lr5e-5', help='experiment name')
    parser.add_argument('--img_downscale', type=int, default=2, help='factor for downscaling the images')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--threshold', type=float, default=0.2, help='threshold for the 2D segmentation results')
    parser.add_argument('--batch_size', type=int, default=7225, help='batch size')
    parser.add_argument('--N_vocab', type=int, default=4000, help='vocabulary size (must be larger than the number of files in training)')
    parser.add_argument('--ckpt_path', type=str, default='./save/ckpts/st_pauls_cathedral/epoch=19.ckpt', help='path for the HaLo-NeRF ckpt after training the RGB head only')
    parser.add_argument('--encode_a', type=bool, default=True, help='whether to encode appearance')
    parser.add_argument('--encode_random', type=bool, default=True, help='whether to encode_random')
    parser.add_argument('--continue_train_semantic', type=bool, default=True, help='whether to continue train the semantic without the RGB')
    parser.add_argument('--enable_semantic', type=bool, default=True, help='whether to enable semantics in the training')
    parser.add_argument('--max_steps', type=int, default=12500, help='max_steps during training')
    parser.add_argument('--semantics_dir', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs/milano/horizontal/clipseg_ft', help='the folder in which the 2D segmentation of the CLIPSeg is saved')
    parser.add_argument('--scene_name', type=str, default='', help='the name of the scene to train')
    parser.add_argument('--save_dir', type=str, default='./sem_results/0_1_undistorted_ft_clip', help='the folder to save the trained model and results')

    # Calculate Metrics
    parser.add_argument('--path_gt', type=str, default='data/manually_gt_masks_st_paul/', help='the path for the ground truth path')
    parser.add_argument('--PRED_THRESHOLD', type=float, default=0.5, help='the threshold that is used for the metrics calculations')

    # Visualization (debugging)
    parser.add_argument('--vis_flag', default=False, action="store_true", help='a flag for visualization for debugging')
    parser.add_argument('--save_training_vis', default=False, action="store_true", help='a flag for save the training images visualizations (for debugging)')


    return parser.parse_args()


def get_k_files(k, csv_path, prompt):
    csv_file = pandas.read_csv(csv_path)
    col = csv_file[prompt]
    col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = csv_file['fn'][files_pos.index]
    pos_confidence_values = files_pos.values.tolist()

    return [names_pos.values.tolist(), pos_confidence_values]


def getCfg(opts, prompt):
    cfg = {}

    # Flags
    cfg['train_HaloNeRF_flag'] = opts.train_HaloNeRF_flag
    cfg['save_for_metric_flag'] = opts.save_for_metric_flag
    cfg['calc_metrics_flag'] = opts.calc_metrics_flag
    cfg['vis_flag'] = opts.vis_flag

    # main
    cfg['root_dir'] = opts.root_dir
    cfg['top_k_files'] = opts.top_k_files
    cfg['csv_path'] = opts.csv_retrieval_path
    cfg['prompt'] = prompt
    cfg['prompt'] = [prompt]

    # HaLo-NeRF Training
    cfg['max_steps'] = opts.max_steps
    cfg['path_images'] = cfg['root_dir'] + '/dense/images'
    cfg['folder2save'] = opts.save_dir.split('/')[-1] + '/clipseg/' + cfg['prompt'][0].replace(' ','_').replace("\'",'') + '/top_' + str(cfg['top_k_files']) + '/'
    cfg['threshold'] = opts.threshold

    if not opts.ignore_csv_use_all_images:
        [cfg['files'], cfg['pos_confidence_values']] = get_k_files(cfg['top_k_files'], cfg['csv_path'], ["score"])
    else:
        cfg['files'] = os.listdir(cfg['path_images'])
        cfg['pos_confidence_values'] = []

    print(cfg['files'])
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
    cfg['files_to_run'] = cfg['files']
    cfg['category'] = prompt
    cfg['semantics_dir'] = os.path.join(opts.semantics_dir, cfg['category'])
    if not os.path.exists(cfg['semantics_dir']):
        raise ValueError('no semantics_dir')
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



