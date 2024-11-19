import streamlit as st
import numpy as np
from PIL import Image

from model_utils import load_model, get_class_names, get_model_metrics
from image_utils import preprocess_image, detect_changes



# Load model and get class names
model = load_model()
class_names = get_class_names()

# App title and description
st.title("Land Use and Land Cover (LULC) Classification")
st.write("Upload one or more satellite images to classify them and detect changes.")

# Display model metrics in sidebar
overall_accuracy, class_accuracies = get_model_metrics()
st.sidebar.title("Model Performance Metrics")
st.sidebar.write(f"Overall Accuracy: {overall_accuracy:.1%}")
st.sidebar.write("Per-class Accuracy:")
for class_name, accuracy in class_accuracies.items():
    st.sidebar.write(f"{class_name}: {accuracy:.2%}")

# File uploader
uploaded_files = st.file_uploader(
    "Choose images...", 
    type=["jpg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    predictions = []
    images = []
    
    # Process each uploaded image
    for uploaded_file in uploaded_files:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            images.append(image)

            # Preprocess and predict
            img_array = preprocess_image(image)
            pred = model.predict(img_array)
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            
            predictions.append(class_idx)
            
            # Display prediction results
            st.write(f"Prediction for {uploaded_file.name}:")
            st.write(f"- Class: **{class_names[class_idx]}**")
            st.write(f"- Confidence: **{confidence:.2%}**")
            
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {e}")

    # Display change detection results
    if len(predictions) > 1:
        st.write("\nChanges Detected:")
        changes = detect_changes(predictions, class_names)
        for change in changes:
            st.write(change)