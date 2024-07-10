import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the trained model
model_filename = 'random_forest_iris_model.joblib'
model = load(model_filename)

# Load the dataset for fitting StandardScaler and PCA
data = pd.read_csv("Iris (1).csv")
X = data.drop(['Id', 'Species'], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Streamlit UI
st.title("Iris Flower Classification")

st.write("""
### Enter the characteristics of the Iris flower
""")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, format="%.1f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, format="%.1f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, format="%.1f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, format="%.1f")

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, classifier, pca, scaler):
    # Convert string labels to numeric indices
    le = LabelEncoder()
    y_numeric = le.fit_transform(y)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    grid_points_pca = pca.transform(grid_points_scaled)
    
    Z = classifier.predict(grid_points_pca)
    Z = le.transform(Z)
    
    # Reshape the predictions to match xx and yy shapes
    Z = Z.reshape(xx.shape)
    
    # Create a figure and plot the decision boundaries
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y_numeric, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundaries")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()


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

    # Plot the decision boundaries
    st.write("### Decision Boundaries Visualization")
    st.pyplot(plot_decision_boundaries(X_pca, data['Species'], model, pca, scaler))


