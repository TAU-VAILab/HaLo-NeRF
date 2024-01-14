from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--prompts', type=str, default='windows', help='all the text prompts to segment, sperated with ;')
    parser.add_argument('--folder_to_save', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs/milano/horizontal', help='path to save the segmentations')
    parser.add_argument('--model_path', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs', help='the path where the fine-tuned clipseg is saved')
    parser.add_argument('--building_type', type=str, default='cathedral', help='the building type on which you whould llke to segment. It can be mosqte/cathedral/synagogue/building/all/etc.')  #'mosque' 'cathedral' 'synagogue' 'building', 'all'
    parser.add_argument('--csv_retrieval_path', type=str, default='data/milano_geometric_occlusions.csv', help='the csv file that containes the score of the files for retreival. The retrieval of the files will be sorted accordingly.')
    parser.add_argument('--images_folder', type=str, default='data/0_1_undistorted/dense/images/', help='the folder that contained the images for segmentation')
    parser.add_argument('--use_csv_for_retrieval', type=bool, default=True, help='If to use the csv for retrieval of the images. You can change to False to run on all images.')
    parser.add_argument('--n_files', type=int, default=150, help='the number of files to retrieve from the csv retrieval')
    parser.add_argument('--save_images', type=bool, default=False, help='save the images to a HTML file for visualization')
    parser.add_argument('--save_baseline', type=bool, default=False, help='save the segmentation of baseline CLIPSeg model')
    parser.add_argument('--save_refined_clipseg', type=bool, default=True, help='save the segmentation of fine-tuned CLIPSeg model')
    return parser.parse_args()