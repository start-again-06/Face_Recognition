# # 😃 Face Recognition using FaceNet and Triplet Loss

This project implements a face recognition system using the **FaceNet** architecture with **Triplet Loss**, trained to identify and verify faces using deep feature embeddings.

---

## 📌 Overview

- **Framework**: Keras + TensorFlow backend
- **Model**: FaceNet using Inception blocks
- **Loss Function**: Triplet Loss
- **Task**: Face Verification & Identification
- **Input Shape**: (3, 96, 96)

---

## 🧠 Core Components

### 🔷 Model Architecture
- Inspired by the [FaceNet](https://arxiv.org/abs/1503.03832) paper
- Inception blocks (v2) for feature extraction
- Generates a 128-dimensional embedding vector per face

### 🔶 Loss Function
- **Triplet Loss**: Encourages the anchor-positive pair to be closer than the anchor-negative pair by a margin `\alpha`

```python
loss = \sum{max(\|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha, 0)}

🧪 Training & Initialization

Triplet loss implemented and tested using TensorFlow sessions

Model compiled with Adam optimizer and custom loss

Pre-trained weights loaded using load_weights_from_FaceNet()

🧾 Usage

✅ Face Verification

Given an image and an identity, the system checks if the person is who they claim to be.

verify("images/camera_0.jpg", "younes", database, FRmodel)

🔍 Face Recognition

Identifies the closest match in the database for a given image.

who_is_it("images/camera_0.jpg", database, FRmodel)

🗃️ Face Database

Images are encoded into 128D vectors and stored in a dictionary with names as keys:

database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)

📊 Output Sample

Face matched correctly with message: It's younes, welcome in!

Face not matched: It's not kian, please go away

## 📚 References

- Florian Schroff, Dmitry Kalenichenko, James Philbin – [FaceNet: A Unified Embedding for Face Recognition and Clustering (2015)](https://arxiv.org/abs/1503.03832)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf – [DeepFace: Closing the gap to human-level performance in face verification (2014)](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- Victor Sy Wang – [Keras-OpenFace GitHub Repository](https://github.com/iwantooxxoox/Keras-OpenFace)
- David Sandberg – [FaceNet GitHub Repository](https://github.com/davidsandberg/facenet)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
