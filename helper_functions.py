import streamlit as st
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

@st.cache_resource
def load_sam_model(model_type):
    models_route = './models/'
    model_name = ''
    device = "cuda"

    if model_type == 'vit_b':
        model_name = 'sam_vit_b_01ec64'
    elif model_type == 'vit_h':
        model_name = 'sam_vit_h_4b8939'
    elif model_type == 'vit_l':
        model_name = 'sam_vit_l_0b3195'

    sam = sam_model_registry[model_type](checkpoint=models_route+model_name+'.pth')
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    return mask_generator, predictor
    
def get_images_list():
    path = './images/'
    files = os.listdir(path)
    extensions = ['png','jpg','jpeg']
    images_list = []

    for image in files:
        if image.split('.')[-1] in extensions:
            images_list.append(image)

    return images_list


def load_image(image_name):
    path = './images/'
    image = cv2.imread(path+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.shape[1]*image.shape[0]
    pixel_limit = 538450
    resized_image = image
    if pixels>pixel_limit:
        scale_percent = round(pixel_limit/pixels,2)*100
        # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    original_image = image
    return original_image, resized_image

def show_anns(anns, image):
    fig = plt.figure()
    fig.set_size_inches(20,20)
    plt.imshow(image)
    if len(anns) == 0:
        return
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    fig.canvas.draw()
    
    b = fig.axes[0].get_window_extent()
    img = np.array(fig.canvas.buffer_rgba())
    img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    return img

def get_masked_df(masks):
    
    l = [(mask['area'], mask['stability_score']) for mask in masks]
    l = np.transpose(np.array(l))
    return pd.DataFrame({'area':l[0], 'score':l[1]})