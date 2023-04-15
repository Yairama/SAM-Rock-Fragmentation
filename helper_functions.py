import streamlit as st
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import gc


PIXEL_LIMIT = 1500**2

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
    # predictor = SamPredictor(sam)

    return mask_generator#, predictor
    
def get_images_list():
    path = './images/'
    files = os.listdir(path)
    extensions = ['png','jpg','jpeg']
    images_list = []

    for image in files:
        if image.split('.')[-1].lower() in extensions:
            images_list.append(image)

    return images_list


def load_image(image_name):
    path = './images/'
    image = cv2.imread(path+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.shape[1]*image.shape[0]
    pixel_limit = PIXEL_LIMIT
    resized_image = image
    # resized_image = cv2.bilateralFilter(resized_image, 11, 75, 75)
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

    fig,ax = plt.subplots()
    fig.set_size_inches(20,20)
    plt.imshow(image)
    if len(anns) == 0:
        return
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

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
    fig.clf()
    ax.clear()
    plt.close()
    gc.collect()

    return img
    
def get_masked_df(masks):
    
    l = [(mask['area'], mask['stability_score']) for mask in masks]
    l = np.transpose(np.array(l))
    return pd.DataFrame({'area':l[0], 'score':l[1]})

def get_balls_data(img):
    image = img.copy()
    balls_data = {'x':[], 'y':[], 'area':[]}

    original = image.copy()

    lower = np.array([240, 0, 0], dtype="uint8")
    upper = np.array([255, 30, 30], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    detected = cv2.bitwise_and(original, original, mask=mask)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and find total area
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area = 0
    for c in cnts:
        area += cv2.contourArea(c)
        cv2.drawContours(original,[c], 0, (0,0,0), 2)


    gray = opening
    
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    # list for storing names of shapes
    for contour in contours:

        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        # using drawContours() function
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 1)
    
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        # cv2.putText(opening, 'circle', (x, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        balls_data['x'].append(x)
        balls_data['y'].append(y)
        balls_data['area'].append(area)

    return pd.DataFrame(balls_data)