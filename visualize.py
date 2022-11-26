import numpy as np
import cv2
import os
from tqdm import tqdm

result_img_path = "./predictions/result_img"
if not os.path.exists(result_img_path):
    os.mkdir(result_img_path)

results = np.load("./predictions/prediction.npy")
image_size = results.shape[0]

for image_num in tqdm(range(image_size)):
    image = results[image_num]
    
    print(image.min())
    print(image.max())
    
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(result_img_path, str(image_num) + ".jpg")
    
    cv2.imwrite(image_path, image)