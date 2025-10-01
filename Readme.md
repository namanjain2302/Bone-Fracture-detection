🦴 Bone Fracture Detection using Deep Learning

This project focuses on automated bone fracture detection from X-ray images using deep learning techniques. It leverages a pre-trained DenseNet-121 model (transfer learning) as a feature extractor and fine-tunes it for medical image classification.

🔍 Key Features

Dataset Processing: Applied advanced preprocessing — grayscale conversion, adaptive histogram equalization, resizing, and normalization for consistent image quality.

Model Architecture: Utilized DenseNet-121 for feature extraction and a custom classification head for binary fracture detection.

Training Pipeline: Implemented data augmentation, early stopping, and learning-rate scheduling to enhance generalization.

UI Deployment: Integrated a Gradio interface for intuitive image upload and real-time fracture prediction.

Performance: Achieved ~69% validation accuracy on the medical dataset.

🧠 Tech Stack

Python · PyTorch · OpenCV · Scikit-learn · Gradio

🚀 Future Improvements

Incorporate Grad-CAM visualization for explainable AI.

Expand dataset for better generalization across bone types.

Deploy the model on Hugging Face Spaces for public access.
