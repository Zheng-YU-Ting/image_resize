import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

IMAGE_SIZE = (128, 128)
n=1

datasets = ['testFER-copy']#資料夾
#datasets = ['trainFER_data-augmentation-Keras', 'testFER_data-augmentation-Keras']

class_names = ["angry","fear","happy","neutral","sad","surprise"]
#class_names = ["surprise"]

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

# Iterate through training and test sets
for dataset in datasets:
    
    print("Loading {}".format(dataset))
    
    # Iterate through each folder corresponding to a category
    for folder in os.listdir(dataset):
        #print(folder)
        label = class_names_label[folder]
        
        # Iterate through each image in our folder
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            # Get the path name of the image
            img_path = os.path.join(os.path.join(dataset, folder), file)
            #print(img_path)
            # Open and resize the img
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #print(image.shape)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #cv讀照片，顏色莫認為BGR，需轉為RGB
            
            image = cv2.resize(image, IMAGE_SIZE)
            
            cv2.imwrite(img_path, image)
            n+=1