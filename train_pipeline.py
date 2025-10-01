import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
from glob import glob
from PIL import Image, UnidentifiedImageError
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse

class GLCMFeatureExtractor:
    def __init__(self, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles
    
    def preprocess_xray(self, img_path):
        """Robust image loading with PIL only"""
        try:
            with Image.open(img_path) as pil_img:
                # Convert to grayscale
                if pil_img.mode != 'L':
                    pil_img = pil_img.convert('L')
                
                # Resize and convert to numpy array
                pil_img = pil_img.resize((256, 256))
                img = np.array(pil_img)
            
            # Handle empty images
            if img.size == 0:
                raise ValueError(f"Empty image: {img_path}")
                
            # Improved normalization
            img = img.astype(np.float32)
            min_val = np.min(img)
            max_val = np.max(img)
            
            # Handle zero-contrast images
            if max_val - min_val < 1e-5:
                img = np.zeros_like(img)  # Return black image
            else:
                img = (img - min_val) / (max_val - min_val) * 255
                
            return img.astype(np.uint8)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def extract_features(self, img):
        """Extract GLCM features with validation"""
        if img is None:
            return None
            
        try:
            # Calculate GLCM with optimized parameters
            glcm = graycomatrix(
                img, 
                distances=self.distances, 
                angles=self.angles, 
                levels=256,
                symmetric=True, 
                normed=True
            )
            
            # Extract texture properties
            features = []
            props = ['contrast', 'dissimilarity', 'homogeneity', 
                    'energy', 'correlation', 'ASM']
            
            for prop in props:
                feat = graycoprops(glcm, prop)
                features.extend(feat.flatten())
                
            return np.array(features)
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def extract_from_folder(self, folder_path, max_samples=None):
        """Batch feature extraction with error handling"""
        features = []
        labels = []
        class_name = os.path.basename(folder_path)
        
        # Find all image files
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.dcm', '*.tif', '*.bmp'):
            image_paths.extend(glob(os.path.join(folder_path, ext)))
            
        if not image_paths:
            print(f"Warning: No images found in {folder_path}")
            return np.array([]), np.array([])
            
        # Apply sampling if requested
        if max_samples and len(image_paths) > max_samples:
            image_paths = np.random.choice(image_paths, max_samples, replace=False)
            
        # Process each image
        for img_path in image_paths:
            img = self.preprocess_xray(img_path)
            if img is None:
                continue
                
            feat = self.extract_features(img)
            if feat is not None:
                features.append(feat)
                labels.append(class_name)
                
        print(f"Successfully processed {len(features)}/{len(image_paths)} images in {folder_path}")
        return np.array(features), np.array(labels)


def load_dataset(dataset_path):
    splits = ['train', 'val', 'test']
    features = {split: [] for split in splits}
    labels = {split: [] for split in splits}
    extractor = GLCMFeatureExtractor()
    
    for split in splits:
        for label in ['fractured', 'not_fractured']:
            folder = os.path.join(dataset_path, split, label)
            if not os.path.exists(folder):
                print(f"Warning: Missing folder {folder}")
                continue
                
            feats, lbls = extractor.extract_from_folder(folder)
            if len(feats) > 0:
                features[split].extend(feats)
                labels[split].extend(lbls)
                print(f"Extracted {len(feats)} samples from {split}/{label}")
            else:
                print(f"No valid samples found in {split}/{label}")
                    
    return features, labels


def train_and_evaluate(features, labels, model_save_path='models'):
    os.makedirs(model_save_path, exist_ok=True)
    
    le = LabelEncoder()
    all_labels = []
    for split in labels:
        all_labels.extend(labels[split])
    le.fit(all_labels)
    
    # Prepare data splits
    X_train = np.array(features['train'])
    y_train = le.transform(labels['train'])
    
    X_val = np.array(features['val'])
    y_val = le.transform(labels['val'])
    
    X_test = np.array(features['test'])
    y_test = le.transform(labels['test'])
    
    # Check data availability
    if len(X_train) == 0:
        raise ValueError("No training data available!")
    
    # Train SVM classifier
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    if len(X_val) > 0:
        y_val_pred = clf.predict(X_val)
        print(classification_report(y_val, y_val_pred, target_names=le.classes_))
        print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Validation Recall: {recall_score(y_val, y_val_pred):.4f}")
    else:
        print("No validation data available")
    
    # Evaluate on test set
    print("\nTest Set Performance:")
    if len(X_test) > 0:
        y_test_pred = clf.predict(X_test)
        print(classification_report(y_test, y_test_pred, target_names=le.classes_))
        print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
    else:
        print("No test data available")
    
    # Save model
    model_path = os.path.join(model_save_path, 'fracture_detection_model.joblib')
    encoder_path = os.path.join(model_save_path, 'label_encoder.joblib')
    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    return clf, le

class FracturePredictor:
    def __init__(self, model_path='models/fracture_detection_model.joblib', 
                 encoder_path='models/label_encoder.joblib'):
        # Verify model paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
            
        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)
        self.extractor = GLCMFeatureExtractor()
    
    def predict(self, img_input, visualize=True, save_path='prediction_result.png'):
        """
        Predict fracture from image input (file path)
        Returns: (label, confidence, visualization_path)
        """
        try:
            # Preprocess image
            img = self.extractor.preprocess_xray(img_input)
            if img is None:
                return "Error: Invalid image", 0.0, None
            
            # Extract features
            feat = self.extractor.extract_features(img)
            if feat is None:
                return "Error: Feature extraction failed", 0.0, None
            
            # Make prediction
            proba = self.model.predict_proba(feat.reshape(1, -1))[0]
            pred = self.model.predict(feat.reshape(1, -1))[0]
            label = self.le.inverse_transform([pred])[0]
            confidence = max(proba)
            
            # Generate visualization
            vis_path = None
            if visualize:
                vis_path = save_path
                self.visualize_prediction(img, label, confidence, proba, save_path)
            
            return label, confidence, vis_path
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Prediction error", 0.0, None
    
    def visualize_prediction(self, img, label, confidence, proba, save_path):
        """Create and save prediction visualization"""
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original Image\nPrediction: {label}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        
        # Probability distribution
        plt.subplot(1, 2, 2)
        colors = ['red' if cls != label else 'green' for cls in self.le.classes_]
        plt.bar(self.le.classes_, proba, color=colors)
        plt.title("Classification Probabilities")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bone Fracture Detection System')
    parser.add_argument('--dataset_path', default='dataset', help='Path to dataset directory')
    parser.add_argument('--model_save_path', default='models', help='Path to save trained models')
    parser.add_argument('--predict_image', default=None, help='Path to image for prediction')
    args = parser.parse_args()
    
    if args.predict_image:
        # Predict mode
        predictor = FracturePredictor(
            model_path=os.path.join(args.model_save_path, 'fracture_detection_model.joblib'),
            encoder_path=os.path.join(args.model_save_path, 'label_encoder.joblib')
        )
        label, confidence, vis_path = predictor.predict(args.predict_image)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        if vis_path:
            print(f"Visualization saved to {vis_path}")
    else:
        # Train mode
        print("Loading dataset and extracting features...")
        features, labels = load_dataset(args.dataset_path)
        print("\nTraining and evaluating model...")
        train_and_evaluate(features, labels, args.model_save_path)
