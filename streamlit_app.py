import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64

# Load the trained model
model_filename = 'random_forest_iris_model.joblib'
model = load(model_filename)

# Load the dataset for fitting StandardScaler and PCA
data = pd.read_csv("static/Iris (1).csv")
X = data.drop(['Id', 'Species'], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Streamlit UI
st.set_page_config(page_title="Irish Flower Classification", page_icon=":cherry_blossom:")
col1, col2, col3 = st.columns([0.1,0.8,0.1])

with col2:
    st.title("Iris Flower Classification")



def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode()
    return base64_str

# Specify the path to your image
image_path = "static/iris.jpg"

# Get the base64 string of the image
base64_image = get_base64_image(image_path)

# Create the HTML to display the image with rounded corners
html_code = f'''
    <div style="display: flex; justify-content: center;">
        <img src="data:image/jpeg;base64,{base64_image}" style="width: 300px; border-radius: 15px;">
    </div>
'''

# Display the image using st.markdown
st.markdown(html_code, unsafe_allow_html=True)

st.write("""
### Enter the characteristics of the Iris flower
""")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, format="%.1f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, format="%.1f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, format="%.1f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, format="%.1f")




# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Apply PCA transformation
    input_data_pca = pca.transform(input_data_scaled)
    
    # Make prediction
    prediction = model.predict(input_data_pca)
    
    # Display the prediction
    st.write(f"The predicted Iris species is: **{prediction[0]}**")

# Load custom CSS from styles.css
with open("static/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add "Made by" message at the bottom right corner
st.markdown("""
    <div class="footer">
        Made by Swarnabha
        <a href="https://github.com/swarnabha-dev/" target="_blank"><img src="https://img.icons8.com/?size=30&id=LoL4bFzqmAa0&format=png&color=000000" style="vertical-align: middle;"></a>
        <a href="https://www.instagram.com/swarnabha_halder/" target="_blank"><img src="https://img.icons8.com/fluent/30/000000/instagram-new.png" style="vertical-align: middle;"></a>
        <a href="https://www.linkedin.com/in/swarnabha-halder-627692254/" target="_blank"><img src="https://img.icons8.com/color/30/000000/linkedin.png" style="vertical-align: middle;"></a>
    </div>
    """, unsafe_allow_html=True)



