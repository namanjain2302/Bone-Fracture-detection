import gradio as gr
import numpy as np
import cv2
import tempfile
import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import predictor class
from src.predict_fracture import FracturePredictor

# Get current script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up from app/ to project root

# CORRECTED MODEL PATHS
MODEL_PATH = 'models/fracture_detection_model.joblib'
ENCODER_PATH = 'models/label_encoder.joblib'
# Debugging output
print(f"Project root: {project_root}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Encoder exists: {os.path.exists(ENCODER_PATH)}")

# Initialize predictor only if files exist
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    predictor = FracturePredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
else:
    print("ERROR: Model files not found. Please run training first.")
    # Provide detailed troubleshooting help
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in models directory: {os.listdir(os.path.join(project_root, 'models'))}")
    exit(1)

def is_xray_image(img):
    """Validate if image is an X-ray using intensity distribution"""
    try:
        if isinstance(img, np.ndarray):
            # Convert to grayscale if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                img_gray = img if len(img.shape) == 2 else img[:, :, 0]
        else:
            # Handle file path
            img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                return False
            img_gray = np.array(img_array)
        
        # Calculate statistics
        mean_intensity = np.mean(img_gray)
        std_intensity = np.std(img_gray)
        
        # X-ray characteristics:
        # - Moderate brightness (not too dark/light)
        # - Reasonable contrast
        is_valid = (20 <= mean_intensity <= 230) and (std_intensity >= 10)
        return is_valid
        
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

def predict_fracture(img):
    """Process uploaded image and return prediction results"""
    try:
        # Step 1: Validate if it's an X-ray
        if not is_xray_image(img):
            return "⚠️ Not an X-ray image", "Upload a valid X-ray", None
        
        # Step 2: Process the image
        if isinstance(img, np.ndarray):
            # Convert to BGR format for OpenCV
            if img.shape[2] == 4:  # RGBA image
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:  # RGB image
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img_bgr)
        else:
            # Already a file path
            tmp_path = img
        
        # Step 3: Get prediction
        label, confidence, vis_path = predictor.predict(tmp_path)
        
        # Step 4: Read visualization
        vis_img = cv2.imread(vis_path)
        if vis_img is not None:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Clean up temporary file
        if isinstance(img, np.ndarray) and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return label, f"{confidence:.4f}", vis_img
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error", "N/A", None

# Create Gradio interface
iface = gr.Interface(
    fn=predict_fracture,
    inputs=gr.Image(label="Upload X-Ray Image"),
    outputs=[
        gr.Label(label="Prediction Result"),
        gr.Textbox(label="Confidence Score"),
        gr.Image(label="Prediction Visualization")
    ],
    title="🦴 Bone Fracture Detection System",
    description="Upload an X-ray image to detect bone fractures using GLCM features and SVM classifier",
    examples=[
        [os.path.join("samples", "fractured_1.jpg")],
        [os.path.join("samples", "normal_1.jpg")]
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Add this line to enable public access
    )
