from PIL import Image
import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

path2save = '/home/cc/students/csguests/chendudai/Thesis/data/'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

path = '/home/cc/students/csguests/chendudai/Thesis/data/notre_dame_front_facade/dense/images/'
images = os.listdir(path)
probs_list = []
img_names = []
for img in images:
    image = Image.open(path + img)
    inputs = processor(text=["a photo of a cathedral's facade"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can t
    probs_list += [probs]
    img_names += [img]

new_list = [[img_names], [probs_list]]
df = pd.DataFrame(new_list)
writer = pd.ExcelWriter('clip.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='welcome', index=False)
writer.save()
