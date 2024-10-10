import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex  # Annoy library
from numpy.linalg import norm
import time


# Load pre-trained data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
# Define dimensions based on the feature list
dimensions = feature_list.shape[1]
# Build and load Annoy index
index = AnnoyIndex(dimensions, metric='euclidean')

for i in range(len(feature_list)):
    index.add_item(i, feature_list[i])

index.build(10)  # 10 trees are used to build the index
# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

# Save uploaded file function
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Recommendation function using Annoy
def recommend(features, index, n=5):
    start_time=time.time()
    indices = index.get_nns_by_vector(features, n)
    end_time = time.time()
    print(f"Time taken by Annoy: {end_time - start_time} seconds")
    return indices

# Resize function for images
def resize_image(img_path, size=(2000,2000)):
    img = Image.open(img_path)
    img = img.resize(size, Image.LANCZOS)
    return img


# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        resized_image = display_image.resize((300, 300))
        st.image(resized_image, caption='Uploaded Image')

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, index)

        # Display recommended images with resizing
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(resize_image(filenames[indices[0]]))
        with col2:
            st.image(resize_image(filenames[indices[1]]))
        with col3:
            st.image(resize_image(filenames[indices[2]]))
        with col4:
            st.image(resize_image(filenames[indices[3]]))
        with col5:
            st.image(resize_image(filenames[indices[4]]))
    else:
        st.header("Some error occurred during file upload")

