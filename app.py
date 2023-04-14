"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import os
from helper_functions import get_images_list, get_masked_df, load_sam_model, load_image, show_anns
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

st.set_page_config(layout="wide")
st.markdown('# Rock fragmentation analysis')
st.markdown('## Decorous edition')


mask_generator, predictor = load_sam_model('vit_b')

image_selected = st.selectbox('Images: ', options=get_images_list(), index=0)


# image = Image.open('./images/'+image_selected)

original_image, resized_image = load_image(image_selected)

st.image(original_image, caption='Original Image')

masks = mask_generator.generate(resized_image)


st.image(show_anns(masks,resized_image), caption='Total masks: '+str(len(masks)))
df_masks = get_masked_df(masks)

col_dataframe, col_stadistics = st.columns(2)
with col_dataframe:
    st.write('Masks Dataframe')
    st.write(df_masks)

with col_stadistics:
    st.write('')
    st.write('Masks Stadistics')
    st.write(df_masks.describe())

fig, ax = plt.subplots()
df_masks['area'].hist(ax=ax)
st.pyplot(fig)