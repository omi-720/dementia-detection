import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import io
import matplotlib.pyplot as plt
import os
import time

# Import the explainer class - make sure explainable_ai.py is in the same directory
try:
    from explainable_ai import DementiaExplainer
except ImportError:
    st.error("Could not import DementiaExplainer class. Make sure explainable_ai.py is in the same directory.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Explainable Dementia Classification",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4b778d;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #28536b;
        margin-bottom: 2rem;
    }
    .result-header {
        font-size: 1.8rem;
        color: #28536b;
        margin-top: 1.5rem;
    }
    .probability-bar {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #f0f5f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .high-prob {
        color: #1e6f5c;
        font-weight: bold;
    }
    .medium-prob {
        color: #ff9a3c;
        font-weight: bold;
    }
    .low-prob {
        color: #aaaaaa;
    }
    .explanation-section {
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        padding-top: 1.5rem;
    }
    .explanation-header {
        font-size: 1.6rem;
        color: #28536b;
        margin-bottom: 1rem;
    }
    .explanation-text {
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the explainer (caching to avoid reloading)
@st.cache_resource
def load_explainer():
    try:
        # Try different model paths
        for model_path in ['dementia_model_best.h5', 'dementia_classification_model.h5']:
            if os.path.exists(model_path):
                return DementiaExplainer(model_path)
        
        # If we get here, try with default paths in the DementiaExplainer class
        return DementiaExplainer()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Title and introduction
st.markdown("<h1 class='main-header'>Explainable Dementia Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Upload a brain scan image to classify dementia stage with explanations</h3>", unsafe_allow_html=True)

# Display information about the model and explainability
with st.expander("About this Model & Explainability"):
    st.markdown("""
    <div class='info-box'>
    <h4>The Model</h4>
    This application uses a Convolutional Neural Network (CNN) trained on MRI brain scans to classify them into four categories:
    
    - <b>Non-Demented</b>: No signs of dementia
    - <b>Mild Demented</b>: Early signs of dementia with mild cognitive impairment
    - <b>Moderate Demented</b>: More pronounced cognitive decline
    - <b>Very Mild Demented</b>: Very early signs of cognitive impairment
    
    <h4>Explainable AI Features</h4>
    
    <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b>: This technique highlights regions in the image that were most influential for the model's classification decision. Red/yellow areas indicate regions the model focused on to make its prediction.
    
    <b>Occlusion Sensitivity</b>: This method systematically blocks different parts of the image to see how the prediction changes. Areas that cause the largest drop in prediction confidence when occluded are highlighted, indicating regions that are most important for classification.
    
    These explainability features help medical professionals understand and trust the model's decision-making process.
    
    <i>Note: This tool is for educational and demonstration purposes only. It should not replace professional medical diagnosis.</i>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Controls")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])

# Explanation method selection
st.sidebar.markdown("## Explanation Method")
explanation_method = st.sidebar.radio(
    "Select explanation method",
    ["No Explanation", "GradCAM", "Occlusion Sensitivity", "Both"]
)

# Occlusion parameters (only shown if occlusion is selected)
if explanation_method in ["Occlusion Sensitivity", "Both"]:
    st.sidebar.markdown("## Occlusion Parameters")
    patch_size = st.sidebar.slider("Patch Size", min_value=8, max_value=48, value=16, step=4)
    stride = st.sidebar.slider("Stride", min_value=4, max_value=24, value=16, step=4)
    st.sidebar.markdown("*Note: Smaller values give more detailed results but take longer*")

# Add disclaimer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Disclaimer:** This app is for educational purposes only and should not be used for medical diagnosis. Always consult healthcare professionals for medical advice.
""")

# Main content area
if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file).convert('RGB')
    
    # Load the explainer
    explainer = load_explainer()
    
    if explainer is None:
        st.error("Could not load the model. Please check that the model file exists and is valid.")
        st.stop()
    
    # Make the prediction
    with st.spinner("Analyzing image..."):
        try:
            pred_class, probabilities = explainer.predict(img)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.stop()
    
    # Display the prediction results
    col1, col2 = st.columns([1, 2])
    
    # Display the uploaded image
    with col1:
        st.markdown("### Uploaded Image")
        st.image(img, width=300, caption="Original Brain Scan")
        
        # Display the predicted class with confidence
        st.markdown("<h2 class='result-header'>Classification Result</h2>", unsafe_allow_html=True)
        display_class = pred_class.replace('_', ' ')
        st.markdown(f"<h3>Predicted Stage: <span style='color:#4b778d'>{display_class}</span></h3>", unsafe_allow_html=True)
        
        # Get the confidence for the predicted class
        confidence = probabilities[pred_class] * 100
        st.markdown(f"<h4>Confidence: <span style='color:#28536b'>{confidence:.2f}%</span></h4>", unsafe_allow_html=True)
    
    # Display probability distribution
    with col2:
        st.markdown("### Probability Distribution")
        
        # Sort probabilities for better visualization
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Display probability bars
        for class_name, prob in sorted_probs:
            # Format the class name for display
            display_name = class_name.replace("_", " ")
            
            # Determine color class based on probability
            if prob > 0.7:
                prob_class = "high-prob"
            elif prob > 0.3:
                prob_class = "medium-prob"
            else:
                prob_class = "low-prob"
            
            # Create the progress bar and label
            prob_percentage = prob * 100
            st.markdown(f"<div class='probability-bar'><span class='{prob_class}'>{display_name}: {prob_percentage:.2f}%</span></div>", unsafe_allow_html=True)
            st.progress(float(prob))
    
    # Generate explanations based on selected method
    if explanation_method != "No Explanation":
        st.markdown("<div class='explanation-section'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='explanation-header'>Explanation of the Model's Decision</h2>", unsafe_allow_html=True)
        
        explanation_cols = st.columns(2 if explanation_method == "Both" else 1)
        
        if explanation_method in ["GradCAM", "Both"]:
            with explanation_cols[0]:
                st.markdown("### Grad-CAM Visualization")
                st.markdown("""
                <div class='explanation-text'>
                This visualization highlights the regions of the brain scan that most influenced the model's classification decision.
                Areas in <span style='color:red'>red/yellow</span> show regions the model focused on.
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating Grad-CAM explanation..."):
                    try:
                        _, _, grad_superimposed, _, _ = explainer.explain_with_gradcam(img)
                        st.image(grad_superimposed, caption=f"Grad-CAM: Important regions for {display_class} classification", width=400)
                    except Exception as e:
                        st.error(f"Error generating Grad-CAM: {e}")
        
        if explanation_method in ["Occlusion Sensitivity", "Both"]:
            with explanation_cols[1 if explanation_method == "Both" else 0]:
                st.markdown("### Occlusion Sensitivity")
                st.markdown(f"""
                <div class='explanation-text'>
                This visualization shows how covering different parts of the image affects the model's confidence.
                Areas in <span style='color:red'>red/yellow</span> significantly decrease the model's confidence when covered.
                <br><i>Parameters: Patch Size = {patch_size}, Stride = {stride}</i>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating occlusion sensitivity map (this may take a while)..."):
                    try:
                        _, _, occ_superimposed, _, _ = explainer.explain_with_occlusion(
                            img, 
                            patch_size=patch_size, 
                            stride=stride
                        )
                        st.image(occ_superimposed, caption=f"Occlusion Sensitivity: Important regions for {display_class} classification", width=400)
                    except Exception as e:
                        st.error(f"Error generating occlusion map: {e}")
        
        # Add clinical interpretation section
        st.markdown("<div class='explanation-section'></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='explanation-header'>Clinical Interpretation</h2>", unsafe_allow_html=True)
        
        # Different interpretation text based on the classification
        if "NonDemented" in pred_class:
            st.markdown("""
            <div class='info-box'>
            <h4>Non-Demented Interpretation</h4>
            <p>The model has classified this brain scan as showing no signs of dementia. Key indicators typically include:</p>
            <ul>
                <li>Normal hippocampal volume (memory center of the brain)</li>
                <li>Normal ventricular size</li>
                <li>No significant cortical atrophy</li>
                <li>Normal gray/white matter contrast</li>
            </ul>
            <p>The highlighted areas in the explanation visualization show the regions the model focused on to make this determination.</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif "MildDemented" in pred_class:
            st.markdown("""
            <div class='info-box'>
            <h4>Mild Dementia Interpretation</h4>
            <p>The model has classified this brain scan as showing mild dementia. Key indicators typically include:</p>
            <ul>
                <li>Mild hippocampal atrophy</li>
                <li>Slight ventricular enlargement</li>
                <li>Early signs of cortical thinning</li>
                <li>Mild widening of sulci in specific regions</li>
            </ul>
            <p>The highlighted areas in the explanation visualization show the regions with subtle changes that the model has detected.</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif "ModerateDemented" in pred_class:
            st.markdown("""
            <div class='info-box'>
            <h4>Moderate Dementia Interpretation</h4>
            <p>The model has classified this brain scan as showing moderate dementia. Key indicators typically include:</p>
            <ul>
                <li>Significant hippocampal volume loss</li>
                <li>Moderate ventricular enlargement</li>
                <li>Pronounced cortical atrophy in temporal and parietal regions</li>
                <li>Clear widening of sulci and loss of brain volume</li>
            </ul>
            <p>The highlighted areas in the explanation visualization indicate regions with substantial changes typical of moderate disease progression.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:  # Very Mild Demented
            st.markdown("""
            <div class='info-box'>
            <h4>Very Mild Dementia Interpretation</h4>
            <p>The model has classified this brain scan as showing very mild dementia. Key indicators typically include:</p>
            <ul>
                <li>Subtle changes in hippocampal morphology</li>
                <li>Minimal ventricular expansion</li>
                <li>Very early cortical thinning difficult to detect visually</li>
                <li>Subtle changes in specific brain regions associated with early pathology</li>
            </ul>
            <p>The highlighted areas in the explanation visualization show regions with very subtle changes that might be missed in visual inspection but were detected by the model.</p>
            </div>
            """, unsafe_allow_html=True)
else:
    # Show placeholder instructions
    st.markdown("""
    ## Instructions
    
    1. Upload a brain MRI scan image using the file uploader in the sidebar
    2. Select an explanation method to understand how the model makes decisions
    3. The model will:
       - Analyze the image and classify it into one of four dementia stages
       - Generate visual explanations showing which parts of the brain scan influenced the decision
    
    The explainable AI features help make the "black box" of deep learning more transparent, showing which features in the brain scan were most important for classification.
    
    Please ensure the uploaded image is a clear brain MRI scan for best results.
    """)