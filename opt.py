import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['blender', 'phototourism'],
                        help='which dataset to train/val')

    # for blender
    parser.add_argument('--data_perturb', nargs="+", type=str, default=[],
                        help='''what perturbation to add to data.
                                Available choices: [], ["color"], ["occ"] or ["color", "occ"]
                             ''')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=15,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    # Ha-NeRF parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--use_mask', default=False, action="store_true",
                        help='whether to use mask')
    parser.add_argument('--encode_random', default=False, action="store_true",
                        help='whether to encode_random')

    # Loss parameters
    parser.add_argument('--maskrs_max', type=float, default=5e-2,
                        help='regularize mask size')
    parser.add_argument('--maskrs_min', type=float, default=6e-3,
                        help='regularize mask size')
    parser.add_argument('--maskrs_k', type=float, default=1e-3,
                        help='regularize mask size')
    parser.add_argument('--maskrd', type=float, default=0,
                        help='regularize mask digit')
    parser.add_argument('--weightKL', type=float, default=1e-5,
                        help='regularize encA')
    parser.add_argument('--weightRecA', type=float, default=1e-3,
                        help='Rec A')
    parser.add_argument('--weightMS', type=float, default=1e-6,
                        help='mode seeking')

    parser.add_argument('--scale_anneal', type=float, default=-1,
                        help='scale_anneal')
    parser.add_argument('--min_scale', type=float, default=0.5,
                        help='min_scale')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--save_dir', type=str, default="./save",
                        help='checkpoint path to save')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='print the progress bar every X steps')

    ## Semantic Nerf
    parser.add_argument('--enable_semantic', default=False, action="store_true",
                        help='whether to enable semantics')
    parser.add_argument('--num_semantic_classes', type=int, default=2,
                        help='The number of semantic classes')

    parser.add_argument('--continue_train_semantic', default=False, action="store_true",
                        help='whether to continue train the semantic without the RGB')

    parser.add_argument('--Train_with_clipseg', default=False, action="store_true",
                        help='whether to continue train the semantic without the RGB')

    parser.add_argument('--use_highlighter_loss', default=False, action="store_true",
                        help='whether to continue train with highlighter loss')
    parser.add_argument('--prompt', type=str, default='a photo of glowing windows with dark background',
                        help='category name')
    parser.add_argument('--semantics_dir', type=str, default= [],
                        help='semantics dir')
    parser.add_argument('--files_to_run', type=str, default=[],
                        help='files_to_run')






    # Flags For SeRF (do not change)
    parser.add_argument('--clipseg_flag', default=False, action="store_true")
    parser.add_argument('--create_aug_flag', default=False, action="store_true")
    parser.add_argument('--use_refined_clipseg', default=False, action="store_true")
    parser.add_argument('--train_SeRF_flag', default=False, action="store_true")
    parser.add_argument('--save_for_metric_flag', default=False, action="store_true")
    parser.add_argument('--calc_metrics_flag', default=False, action="store_true")
    parser.add_argument('--vis_flag', default=False, action="store_true")
    parser.add_argument('--prompts', type=str, default="spires;window;portal;facade")  # spires;window;portal;facade
    parser.add_argument('--top_k_files', type=int, default=15)
    parser.add_argument('--xls_path', type=str,
                        default='/home/cc/students/csguests/chendudai/Thesis/data/ft_clip_sims_v0.2-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24.csv')  #
    parser.add_argument('--use_rgb_loss', type=bool, default=False)
    parser.add_argument('--path_gt', type=str,
                        default='/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_0_1/')  #
    parser.add_argument('--save_training_vis', default=False, action="store_true")
    parser.add_argument('--neg_prec', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--use_threshold', default=False, action="store_true")
    parser.add_argument('--threshold', type=float, default=0.24)
    parser.add_argument('--vis_prompt_path', type=str, default='/home/cc/students/csguests/chendudai/Thesis/data/visual_prompts_v1.pk')
    parser.add_argument('--use_vis_prompt', default=False, action="store_true")
    parser.add_argument('--neg_files', type=str, default=[])
    parser.add_argument('--use_semantic_function', type=str, default='')
    parser.add_argument('--scene_name', type=str, default='')
    parser.add_argument('--in_server', type=str, default='storage')   #'/home/cc/students/csguests/chendudai/Thesis/data/ft_clip_sims_v0.3-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24-nodups.csv' #retrieval_clip_outdoor_020523.csv
    parser.add_argument('--max_steps', type=int, default=12500000, help='max_steps during training')

    return parser.parse_args()
