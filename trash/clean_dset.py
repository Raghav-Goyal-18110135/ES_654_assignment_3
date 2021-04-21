from os import listdir,remove
from numpy import asarray
from numpy import save
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

for x in listdir("../q7_img"):
    for z in listdir("../q7_img/"+x):
        ind=0
        for y in listdir("../q7_img/"+x+'/'+z):
            ind+=1
            p="../q7_img/"+x+'/'+z+'/'+y
            print(p)
            arr = load_img(p, target_size=(224, 224))
            arr = img_to_array(arr).astype(np.uint8)
            print(arr)
            im = Image.fromarray(arr)
            remove(p)
            im.save("../q7_img/"+x+'/'+z+'/'+str(ind)+".jpg")


        