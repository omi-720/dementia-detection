import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import os

class DementiaExplainer:
    def __init__(self, model_path='dementia_model_best.h5'):
        """Initialize the explainer with a trained model"""
        # Try multiple possible model file names
        possible_models = [
            model_path,
            'dementia_classification_model.h5',
            'dementia_model_best.h5'
        ]
        
        # Try to load the model from any of the possible paths
        model_loaded = False
        for model_file in possible_models:
            try:
                self.model = load_model(model_file)
                print(f"Successfully loaded model from {model_file}")
                model_loaded = True
                break
            except (OSError, IOError) as e:
                print(f"Could not load model from {model_file}: {e}")
                continue
        
        if not model_loaded:
            raise FileNotFoundError("Could not find any valid model file")
        
        self.img_size = 128  # This should match your model's input size
        self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        
        # Initialize the model with a dummy input
        dummy_input = np.zeros((1, self.img_size, self.img_size, 3))
        _ = self.model(dummy_input)
        
        # Create a gradients model for GradCAM
        self.grad_model = self._make_gradcam_model()
    
    def _make_gradcam_model(self):
        """Create a model that outputs both predictions and gradients for GradCAM"""
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer in the model")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.output, last_conv_layer.output]
        )
        
        return grad_model
    
    def preprocess_image(self, img):
        """Preprocess an image for the model"""
        # Load the image if it's a path
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
            
        # Ensure image is a PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize the image
        img = img.resize((self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        return img_array, img
    
    def predict(self, img):
        """Make a prediction for an image"""
        img_array, _ = self.preprocess_image(img)
        predictions = self.model.predict(img_array)
        pred_class_idx = np.argmax(predictions[0])
        pred_class = self.class_names[pred_class_idx]
        probabilities = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        
        return pred_class, probabilities
    
    def explain_with_gradcam(self, img):
        """Generate a Grad-CAM heatmap for an image"""
        img_array, original_img = self.preprocess_image(img)
        
        # Get the model's prediction
        predictions = self.model.predict(img_array)
        pred_class_idx = np.argmax(predictions[0])
        pred_class = self.class_names[pred_class_idx]
        probabilities = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        
        # Run gradient model
        with tf.GradientTape() as tape:
            preds, conv_outputs = self.grad_model(img_array)
            class_output = preds[:, pred_class_idx]
        
        # Calculate gradients
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Create heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        # Process the visualization
        original_img_array = np.array(original_img)
        heatmap = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap_colored * 0.4 + original_img_array
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        return original_img_array, heatmap, superimposed_img, pred_class, probabilities
    
    def explain_with_occlusion(self, img, patch_size=24, stride=12):
        """Generate an occlusion sensitivity map"""
        img_array, original_img = self.preprocess_image(img)
        original_img_array = np.array(original_img)
        
        # Get original prediction
        predictions = self.model.predict(img_array)
        pred_class_idx = np.argmax(predictions[0])
        original_score = predictions[0][pred_class_idx]
        
        # Create sensitivity map
        h, w = self.img_size, self.img_size
        sensitivity_map = np.zeros((h, w))
        
        # Apply occlusion patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Create occluded image
                occluded_img = img_array.copy()
                occluded_img[0, y:y+patch_size, x:x+patch_size, :] = 0.5
                
                # Get prediction
                occluded_predictions = self.model.predict(occluded_img)
                occluded_score = occluded_predictions[0][pred_class_idx]
                
                # Calculate difference
                diff = original_score - occluded_score
                sensitivity_map[y:y+patch_size, x:x+patch_size] += diff
        
        # Process the map
        sensitivity_map = np.maximum(sensitivity_map, 0)
        if np.max(sensitivity_map) > 0:
            sensitivity_map = sensitivity_map / np.max(sensitivity_map)
        
        # Create visualization
        occlusion_heatmap = cv2.resize(sensitivity_map, (original_img_array.shape[1], original_img_array.shape[0]))
        heatmap_colored = np.uint8(255 * occlusion_heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Superimpose on original image
        superimposed_img = heatmap_colored * 0.4 + original_img_array
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        pred_class = self.class_names[pred_class_idx]
        probabilities = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        
        return original_img_array, occlusion_heatmap, superimposed_img, pred_class, probabilities

# Simple test to verify the class works
if __name__ == "__main__":
    try:
        explainer = DementiaExplainer()
        print("DementiaExplainer initialized successfully!")
        
        # Try to find a test image
        test_image_found = False
        for class_dir in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
            path = os.path.join('dementia_dataset', class_dir)
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        test_image = os.path.join(path, file)
                        test_image_found = True
                        break
            if test_image_found:
                break
        
        if test_image_found:
            print(f"Testing with image: {test_image}")
            pred_class, probs = explainer.predict(test_image)
            print(f"Predicted class: {pred_class}")
            
            # Generate GradCAM visualization
            print("Generating GradCAM visualization...")
            _, _, grad_superimposed, _, _ = explainer.explain_with_gradcam(test_image)
            
            # Save the result
            os.makedirs('explanation_results', exist_ok=True)
            Image.fromarray(grad_superimposed).save('explanation_results/gradcam_result.jpg')
            print("GradCAM visualization saved to explanation_results/gradcam_result.jpg")
            
            print("Test completed successfully!")
        else:
            print("No test images found in the dataset directories.")
    except Exception as e:
        print(f"ERROR: {e}")