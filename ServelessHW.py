#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor


# In[2]:


get_ipython().system('wget https://github.com/alexeygrigorev/large-datasets/releases/download/dogs-cats-model/dogs_cats_10_0.687.h5')


# In[3]:


model = keras.models.load_model('dogs_cats_10_0.687.h5')


# In[4]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('dogs_cats_10_0.687.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[5]:


get_ipython().system('ls -lh')


# In[8]:


interpreter = tflite.Interpreter(model_path='dogs_cats_10_0.687.tflite')
interpreter.allocate_tensors()


# In[9]:


interpreter.get_input_details()[0]['index']


# In[71]:


input_index = interpreter.get_input_details()[0]['index']


# In[10]:


interpreter.get_output_details()[0]['index']


# In[72]:


output_index = interpreter.get_output_details()[0]['index']


# In[12]:


# image size from last week assignment: 150 by 150  target should be of the same size


# In[24]:


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# In[25]:


url="https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"


# In[26]:


download_image(url)


# In[27]:


img = download_image(url)


# In[34]:


prepare_image(img, target_size=(150,150))


# In[35]:


resized_img =prepare_image(img, target_size=(150,150))


# In[36]:


np.array(resized_img)


# In[84]:


create_preprocessor('xception', target_size=(150, 150))


# In[85]:


preprocessor= create_preprocessor('xception', target_size=(150, 150))


# In[86]:


X = preprocessor.from_url(url)


# In[87]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()


# In[88]:


interpreter.get_tensor(output_index)


# In[89]:


X[0,0,0,0]


# In[ ]:




