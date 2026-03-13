## Emotion Recognition for Intense Autism Spectrum Disorder Children

This project presents a **Multimodal Emotion Recognition System** designed to identify emotional states in children with Intense Autism Spectrum Disorder (ASD).
Children with ASD often express emotions in atypical ways, making traditional facial emotion recognition systems less reliable. To address this challenge, the proposed system combines facial expressions and hand gestures to improve emotion detection accuracy.
The system uses deep learning with multimodal feature fusion to capture subtle emotional cues and provides real-time emotion prediction through a Streamlit web application.
This project aims to assist clinicians, therapists, and caregivers by providing a system that can detect emotions when ASD children cannot verbally communicate their feelings.

## Objectives

- Develop a multimodal deep learning framework for emotion recognition.
- Extract deep features from facial images and hand gestures using VGG-19.
- Combine both modalities using feature-level fusion.
- Reduce redundant features using a Sparse Autoencoder.
- Classify emotions using a deep neural network with Softmax classifier.
- Deploy the system as a real-time web application using Streamlit.
## Proposed Methodology

**The system follows a multimodal deep learning pipeline:**


1.Facial image

2.Hand gesture image

3.Feature Extraction.
 
4.Pretrained VGG-19 CNN extracts deep features from both images.

5.Feature Fusion.

6.Facial and hand features are combined using feature-level concatenation.

7.Standard Scaler Normalization.

8.A Sparse Autoencoder compresses high-dimensional features into a compact representation.

9.Emotion Classification.

10.Dense layers + Softmax classifier predict the final emotion.

## Dataset Description 
**Facial Emotion Dataset**:Contains images of ASD children with the following emotions:

- Joy
- Sadness
- Anger
- Neutral
- Fear
- Surprise

Images were:
Resized to 224 × 224
Normalized for CNN input
Collected under varying lighting and backgrounds.

**Hand Gesture Dataset**:Hand gesture images representing emotional cues:

- Fist
- Five
- Okay
- Straight
- Peace
- Thumbs

Images were also:
Resized to 224 × 224
Normalized before feature extraction.
## Data Preprocessing

**Steps performed:**

1.Data cleaning (removing noisy and duplicate images)

2.Face and hand region extraction

3.Image resizing and normalization

4.Dataset balancing using upsampling

5.Splitting into training and testing datasets

## Model Comparison

**Baseline Models**:

1️⃣ Facial Expression Only CNN

2️⃣ Hand Gesture Only CNN

3️⃣ Multimodal Fusion without Autoencoder

**Proposed Model:**

**Multimodal Fusion + Sparse Autoencoder + CNN**

**Advantages:**

- Higher accuracy
- Reduced feature redundancy
- Better emotional representation
- Improved generalization

**Deployment**

The trained model is deployed using Streamlit for real-time prediction.

**Users can upload**:

A face image

A hand gesture image

**The system predicts**:

Emotion category

Confidence score

**Live Demo**

**https://emotion-recognition-for-asd-children.streamlit.app/**

**Technologies Used**:Python,Deep Learning,TensorFlow,Keras,VGG-19 ,Sparse Autoencoder.

**Libraries**:NumPy,Scikit-learn.

**Deployment**

Streamlit
