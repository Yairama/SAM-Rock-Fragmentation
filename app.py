"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import os
from helper_functions import df_fixer, get_balls_data, get_images_list, get_masked_df, load_sam_model, load_image, show_anns
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


st.markdown('# Rock fragmentation analysis')
st.markdown('## semi-Decorous edition')


mask_generator= load_sam_model('vit_b')


col_image_selector, col_diameter_imput, col_units = st.columns(3)

with col_image_selector:
    image_selected = st.selectbox('Images: ', options=get_images_list(), index=0)

with col_diameter_imput:
    diameter = st.number_input("Insert Ball's diameter", min_value=1.0, value=10.0, step=1.0)

with col_units:
    units = st.selectbox('Units: ', options=['cm', 'in'], index=0)

original_image, resized_image = load_image(image_selected)
st.image(original_image, caption='Original Image')

area_ball = math.pi*diameter*diameter/4.0

masks = mask_generator.generate(resized_image)
df_masks = get_masked_df(masks)
beta_df_balls = get_balls_data(resized_image)

df_balls_data = df_fixer(beta_df_balls, diameter, beta_df_balls)
df_masks = df_fixer(df_masks, diameter, beta_df_balls)

# with col_balls_data:
#     st.write('Balls data')
#     st.write(df_balls_data)


st.write('Masks Dataframe')
st.write(df_masks)

st.write('')
st.write('Masks Stadistics')
st.write(df_masks.describe())

plot_fig = show_anns(masks,resized_image, df_masks['area_pixel'].quantile(0.8))
st.image(plot_fig, caption='Masked Image')

fig, ax = plt.subplots()
df_masks['area'].hist(bins=100,ax=ax)

ax.axvline(x=df_masks['area'].quantile(0.8), c='r', label='P80')
ax.set_ylabel(f'Area {units}^2')
ax.set_xlabel('Number of Fragments')
plt.legend()

plt.title('Area Histogram, P80: ' + str(round(df_masks['area'].quantile(0.8),1)) + f'{units}^2')
st.pyplot(fig)


fig, ax = plt.subplots()
df_masks['diameter'].hist(bins=100,ax=ax)
ax.axvline(x=df_masks['diameter'].quantile(0.8), c='r', label='P80')
ax.set_ylabel(f'Diameter {units}')
ax.set_xlabel('Number of Fragments')
plt.legend()

plt.title('Diameter Histogram, P80: ' + str(round(df_masks['diameter'].quantile(0.8),1)) + f'{units}')
st.pyplot(fig)