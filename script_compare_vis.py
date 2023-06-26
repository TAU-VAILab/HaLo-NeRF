import os
import glob
import base64

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


ds = 2
load_dir1 = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/sem_results/0_1_undistorted_ft_clip_top50_ds2_ep2_lr5-5/results/phototourism/vis/top_50_nEpochs2/'
load_dir2 = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/sem_results/0_1_undistorted_ft_clip_top100_ds2_ep2_lr5-5_clipsegRefined/results/phototourism/vis/top_100_nEpochs2/'

# save_dir = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/sem_results/test4/clipseg/'
save_dir = load_dir1

html_out = open(os.path.join(save_dir, "clipseg_top50_vs_top100RefinedClipseg.html"), "w")
print('<head><meta charset="UTF-8"></head>', file=html_out)
print("<h1>Results</h1>", file=html_out)

list_prompts = ['window', 'portal', 'spires', 'facade']
for p in list_prompts:

    load_dir_p1 = os.path.join(load_dir1,p + f'_ds{ds}','test')
    load_dir_p2 = os.path.join(load_dir2,p + f'_ds{ds}','test')

    image_paths1 = glob.glob(os.path.join(load_dir_p1, "*.png"))
    image_paths2 = glob.glob(os.path.join(load_dir_p2, "*.png"))

    for image_path1 in image_paths1:
        for image_path2 in image_paths2:
            if os.path.basename(image_path1) == os.path.basename(image_path2):
                print(f"<br><b>{os.path.basename(image_path1)}</b><br>", file=html_out)
                print(f"<br><b>{load_dir1.split('/')[10]}</b><br>", file=html_out)
                print_img(image_path1, html_out)
                print(f"<br><b>{load_dir2.split('/')[10]}</b><br>", file=html_out)
                print_img(image_path2, html_out)

print("<hr>", file=html_out)
html_out.close()

