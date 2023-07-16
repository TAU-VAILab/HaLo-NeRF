from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--prompts', type=str, default='windows')
    parser.add_argument('--folder_to_save', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs/milano/horizontal')
    parser.add_argument('--model_path', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs')
    parser.add_argument('--building_type', type=str, default='cathedral')  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--csv_retrieval_path', type=str, default='data/milano_geometric_occlusions.csv')
    parser.add_argument('--images_folder', type=str, default='data/0_1_undistorted/dense/images/')
    parser.add_argument('--is_geo_occ', type=bool, default=True)
    parser.add_argument('--n_files', type=int, default=150)
    parser.add_argument('--save_images', type=bool, default=False)
    parser.add_argument('--save_baseline', type=bool, default=False)
    parser.add_argument('--save_refined_clipseg', type=bool, default=True)
    return parser.parse_args()