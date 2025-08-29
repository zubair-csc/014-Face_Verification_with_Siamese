# 🔍 014-Face_Verification_with_Siamese

## 📋 Project Overview
This project implements a face verification system in Python using Siamese and FaceNet models. It supports synthetic/real face datasets, face detection, and comprehensive testing. Ideal for computer vision enthusiasts exploring face recognition with TensorFlow.

## 🎯 Objectives
- Implement Siamese and FaceNet models for face verification
- Support synthetic and real face datasets (e.g., LFW)
- Provide face detection and preprocessing
- Achieve high verification accuracy (~90% on synthetic data)
- Offer visualizations for model performance
- Ensure modular, well-documented code

## 📊 Dataset Information
**Datasets**: Synthetic Data, Real Face Datasets (e.g., LFW)
- **Synthetic Data**: 
  - Size: Configurable (e.g., 10 identities, 20 images each)
  - Dimensions: 224x224x3 (RGB)
- **Real Datasets (LFW)**:
  - Size: ~1.3GB, 13,000+ images, 5,000+ identities
  - Dimensions: Variable (resized to 224x224 or 160x160)
- **Target Output**: Binary classification (same/different person)
- **Techniques**: Siamese networks, triplet loss, data augmentation

## 🔧 Technical Implementation
### 📌 Pipeline Architecture
- Siamese Network: Twin CNNs with shared weights, L1 distance
- FaceNet: Inception-based model with triplet loss
- Face detection using Haar cascades
- Modular classes for model training, testing, and inference

### 🧹 Data Preprocessing
- Face detection and cropping
- Resizing to 224x224 (Siamese) or 160x160 (FaceNet)
- Normalization (0-1 range)
- Synthetic data generation with noise and transformations

### ⚙️ Training Configuration
- **Optimizer**: Adam
- **Loss**: Binary cross-entropy (Siamese), triplet loss (FaceNet)
- **Regularization**: Dropout, L2 normalization
- **Hardware**: CPU/GPU support

### 📏 Evaluation Metrics
- Accuracy, precision, recall, F1-score
- ROC curve and AUC
- Embedding quality metrics (intra/inter-class distances)
- Inference time

### 📊 Visualizations
- Confusion matrix
- ROC curve
- Similarity score distribution
- Threshold optimization plots

## 🚀 Getting Started
### Prerequisites
- Python 3.8+, TensorFlow, OpenCV, scikit-learn, matplotlib
- Haar cascade XML file for face detection

### Installation
Clone the repository:
```bash
git clone https://github.com/zubair-csc/014-Face_Verification_with_Siamese.git
cd 014-Face_Verification_with_Siamese
```

### Running the Code
Run the main script:
```python
# Example usage
face_recognition = FaceRecognitionSystem(model_type='siamese')
tester = ModelTester(face_recognition)
images, labels = tester.generate_synthetic_data(num_identities=10, images_per_identity=20)
test_pairs_1, test_pairs_2, test_labels = tester.create_test_pairs(images, labels, 100, 100)
results = tester.test_verification_accuracy(test_pairs_1, test_pairs_2, test_labels)
tester.plot_results('results.png')
```

## 📈 Results
- **Synthetic Data**: ~90% accuracy
- **LFW Dataset**: ~85-90% accuracy (varies by configuration)
- **Model Size**: ~50-100MB
- **Inference Time**: ~0.1-0.5s per pair (CPU)

## 🙌 Acknowledgments
- TensorFlow/Keras for deep learning framework
- OpenCV for face detection
- LFW dataset for real-world face verification testing
