from PIL import Image
import os
import numpy as np

def convert_image(filename,folder2save, f):
    # Open the image in black and white mode
    image = Image.open(filename).convert("L")

    # Create a new image with the same size and mode as the original image
    converted_image = Image.new("RGB", image.size)
    #
    # z = np.array(converted_image)
    # z[z<10] = [255, 0, 0]
    # z[z>=10] = [0, 0, 255]


    # Iterate over each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            pixel_value = image.getpixel((x, y))
            if pixel_value <= 100:  # Black pixel
                converted_image.putpixel((x, y), (255, 0, 0))  # Convert to red
            else:  # White pixel
                converted_image.putpixel((x, y), (0, 0, 255))  # Convert to blue

    # Save the converted image
    os.makedirs(folder2save, exist_ok=True)
    converted_image.save(os.path.join(folder2save,f))



# Example usage
land = ['st_paul', 'hurba', 'notre_dame', 'badshahi', '0209', '0_1']
cat = ['towers', 'windows', 'portals', 'domes', 'minarets', 'spires']
for l in land:
    for c in cat:
        folder = "/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_" + l +'/' + c
        folder2save = '/home/cc/students/csguests/chendudai/Thesis/data/GT_red_and_blue/'+ l +'/' + c
        try:
            l_dir = os.listdir(folder)
        except:
            continue
        for f in l_dir:
            if f.endswith('.jpg'):
                convert_image(os.path.join(folder,f),folder2save, f)
print("Image converted successfully.")