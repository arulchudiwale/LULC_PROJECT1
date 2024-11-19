import tensorflow as tf
import numpy as np
import streamlit as st
import os

def load_model(model_path='lulc_model.h5'):
    """Load and return the trained model."""
    if not os.path.exists(model_path):
        st.error(f"The trained model '{model_path}' was not found. Please check your path.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def get_class_names():
    """Return the list of class names."""
    return [
        'Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
        'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea & Lake'
    ]

def get_model_metrics():
    """Return model performance metrics."""
    # TODO: Replace with actual metrics from model evaluation
    overall_accuracy = 0.935
    class_accuracies = {
        name: np.random.uniform(0.89, 0.98) 
        for name in get_class_names()
    }
    return overall_accuracy, class_accuracies