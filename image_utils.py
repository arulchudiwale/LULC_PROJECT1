import numpy as np
from PIL import Image
import streamlit as st

def preprocess_image(image):
    """Preprocess image for model prediction."""
    try:
        # Convert to RGB and resize
        image = image.convert('RGB')
        image = image.resize((64, 64))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Verify shape
        if img_array.shape != (1, 64, 64, 3):
            st.error("Image preprocessing failed. Please ensure the image size is correct.")
            st.stop()
            
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        st.stop()

def detect_changes(predictions, class_names):
    """Detect and format changes between consecutive predictions."""
    changes = []
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i - 1]:
            change = f"Change between image {i} and image {i+1}: "\
                    f"**{class_names[predictions[i-1]]}** to **{class_names[predictions[i]]}**"
        else:
            change = f"No change detected between image {i} and image {i+1}."
        changes.append(change)
    return changes