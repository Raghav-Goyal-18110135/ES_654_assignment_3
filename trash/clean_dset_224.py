from os import listdir,remove
from numpy import asarray
from numpy import save
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

for x in listdir("../q7_img_224"):
    for z in listdir("../q7_img_224/"+x):
        ind=0
        m = listdir("../q7_img_224/"+x+'/'+z)
        for y in m:
            ind+=1
            p="../q7_img_224/"+x+'/'+z+'/'+y
            print(p)
            arr = load_img(p, target_size=(224, 224))
            arr = img_to_array(arr).astype(np.uint8)
            # print(arr)
            im = Image.fromarray(arr)
            # remove(p)
            im.save(p)
            # im.save("../q7_img_224/"+x+'/'+z+'/'+str(ind)+".jpg")


        