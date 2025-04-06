# Enhancing Generalized Hand Hygiene Recognition


## Objectives

**Abstract**

This project presents the development of a generalized hand hygiene recognition system designed for real-time monitoring using a single-camera setup. Motivated by the global impact of infectious diseases and the critical importance of proper handwashing, the system aims to improve hygiene compliance across diverse users and environments. Leveraging a dataset of WHO-compliant handwashing sequences, the project employs MediaPipe for hand landmark detection and a Long Short-Term Memory (LSTM) model to capture temporal dependencies in hand gestures. The processed data undergo normalization and is reshaped for sequential model input, achieving a recognition accuracy of 93.89%. The system is deployed on an edge device (Jetson Nano) for live inference, offering immediate visual feedback during handwashing. Experimental results demonstrate robust generalization, while limitations related to environmental variability and real-time latency are discussed. Future work includes optimizing performance, expanding datasets, and integrating additional sensors to enhance recognition capabilities and practical deployment.

**Keywords:** Hand Hygiene, WHO Handwashing Steps, Hand Gesture Recognition, MediaPipe, LSTM, Real-Time Monitoring

---

## Credit

- Inspiration from [SunnySideUp11/Hand-Hygiene-ICU](https://github.com/SunnySideUp11/Hand-Hygiene-ICU.git)
- Original dataset Hand Wash Dataset: [https://www.kaggle.com/datasets/realtimear/hand-wash-dataset](https://www.kaggle.com/datasets/realtimear/hand-wash-dataset)

---

## Data Structure

```plaintext
.
├── code
│   └── ...
└── data
    ├── image
    │   ├── 12 folders (one for each handwashing step)
    ├── landmark
    │   ├── 12 CSV files (one for each handwashing step)
    └── HandWashDataset
        ├── 12 folders
        └── multiple video files (one for each handwashing session)
```

---

## Installation

1. Create virtual environment and install libraries:

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn joblib tensorflow seaborn matplotlib
```

2. Clone the repository:

```bash
git clone https://github.com/S-Kia/Hand_washing_application.git
```

---

## Usage

1. Download dataset from Kaggle repository and place it according to the described data structure.
2. Open `Prepare_data.ipynb` to create CSV files (optional).
3. Open `Hand_washing_train.py` to train the model (optional).
4. Deploy `Hand_washing_application.py`, `X.pkl`, and `generalized_lstm_model.keras`.
5. Change camera index if needed:

```python
cap = cv2.VideoCapture(0)
```

---

## The Final Result

- Accuracy: 93.89%
- F1 Score: 93.29%
- Recall: 93.17%

*Add image here*

---

## Limitations

1. **Environmental Variations:** Issues such as motion blur and soap interference cannot be detected by MediaPipe.
2. **Real-Time Response:** The webcam application may respond slowly because it normalizes data, and both MediaPipe and LSTM predict in real time.