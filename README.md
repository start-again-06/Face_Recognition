Face Recognition using FaceNet and Triplet Loss  
A comprehensive computer vision project that implements a face recognition and verification system using the FaceNet architecture with Triplet Loss. The project focuses on learning discriminative facial embeddings that enable accurate face verification and identification through distance-based comparison.

## Features
- End-to-end face recognition and verification pipeline  
- Deep feature embedding generation using FaceNet  
- Triplet Loss for metric learning and identity separation  
- Face verification against claimed identities  
- Face recognition via nearest-neighbor matching  
- Modular and educational implementation  

## Model & Framework
- Model: FaceNet (Inception-based architecture)  
- Framework: TensorFlow with Keras backend  
- Task: Face Verification and Face Identification  
- Input Shape: 96 × 96 × 3 (RGB)  
- Output: 128-dimensional face embedding vector  

## Core Components
- FaceNet model built using Inception v2 blocks  
- Embedding generation for each input face image  
- Triplet Loss function to enforce embedding separation  
- Custom verification and recognition utilities  

## Loss Function
Triplet Loss minimizes the distance between anchor and positive embeddings while maximizing the distance from negative embeddings by a margin α:  
max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + α, 0)


## Training & Initialization
- Triplet Loss implemented and validated using TensorFlow sessions  
- Model compiled with Adam optimizer and custom loss  
- Pretrained FaceNet weights loaded for faster convergence  

## Face Database
- Face images encoded into 128D embeddings  
- Stored as a dictionary mapping identity names to embeddings  
- Used for both verification and recognition tasks  

## Usage
**Face Verification**  
Confirms whether an image matches a claimed identity  

**Face Recognition**  
Identifies the closest matching identity from the database  

## Output
- Successful identity match with confirmation message  
- Rejection message for mismatched identities  

## Dependencies
- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- OpenCV / PIL  

## References
- FaceNet: A Unified Embedding for Face Recognition and Clustering – Schroff et al.  
- DeepFace: Closing the Gap to Human-Level Performance – Taigman et al.  
- Keras-OpenFace Repository  
- FaceNet GitHub Repository  
- Keras Documentation  
- TensorFlow Documentation  

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.


## Loss Function
Triplet Loss minimizes the distance between anchor and positive embeddings while maximizing the distance from negative embeddings by a margin α:  
