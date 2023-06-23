

import os
from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
import numpy as np
from scipy import ndimage

path = 'Dataset/InfectedFish'

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,shear_range=0.15,
                             zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            try:
                image = load_img(root+"/"+directory[j])
                image = img_to_array(image)
                image = image.reshape((1, ) + image.shape)  
                datagen.fit(image)
                for x, val in zip(datagen.flow(image, save_to_dir='aug/'+name, save_prefix='aug', save_format='png'),range(10)):
                    pass
            except Exception as e:
                print(e)    
            
