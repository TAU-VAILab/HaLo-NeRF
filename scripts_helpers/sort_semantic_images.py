from PIL import Image
import os

'''
This script goal is to find the best real images of the scene that describe the semantic area. In order to do that we find
the images that the sum of the pixels value relative to the image area is the highest. This value is the score of the image.
We sort the images according to there score.
'''


# Function to calculate the score of an image
def calculate_score(image_path, threshold=0.5):
    # Open the image using PIL
    img = Image.open(image_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Get the dimensions of the image
    width, height = img.size

    # Count the number of pixels above the threshold
    # above_threshold = sum(1 for pixel_value in img.getdata() if pixel_value > threshold * 255)
    pixel_sum = sum(img.getdata())

    # Calculate the score (ratio of pixels above threshold to image area)
    score = pixel_sum / (width * height)

    return score



#TODO: Change to the directory containing the images
image_directory = 'save/results/phototourism/the_libery_torch/'


# List of image file names in the directory
image_files = os.listdir(image_directory)

# Calculate and store scores for each image
image_scores = {}
for image_file in image_files:
    if image_file.endswith('_semantic.png'):
        image_path = os.path.join(image_directory, image_file)
        score = calculate_score(image_path)
        image_scores[image_file] = score

# Sort the images by their scores in descending order
sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)

# Save the sorted images to a text file
output_file = os.path.join(image_directory, 'sorted_images_sum.txt')
with open(output_file, 'w') as file:
    for image_file, score in sorted_images:
        file.write(f"Image: {image_file}, Score: {score}\n")

print(f"Sorted images list saved to {output_file}")