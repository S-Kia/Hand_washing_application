import os
import mediapipe as mp
import numpy as np
import cv2
import joblib
import time
import threading
import queue
from collections import deque
from tensorflow.keras.models import load_model

# Step names corresponding to predictions
step_names = ["1", "2lt", "2rt", "3", "4lt", "4rt", "5lt", "5rt", "6lt", "6rt", "7lt", "7rt"]

# Load the pre-trained model
model = load_model('generalized_lstm_model.keras')

# Load dataset for normalization
X_train = joblib.load('X.pkl')
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

# Initialize MediaPipe Hands once
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Manual Timer Class
class ManualTimer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.monotonic()
            self.running = True

    def pause(self):
        if self.running:
            self.elapsed += time.monotonic() - self.start_time
            self.running = False

    def resume(self):
        if not self.running:
            self.start_time = time.monotonic()
            self.running = True

    def reset(self):
        self.start_time = None
        self.elapsed = 0
        self.running = False

    def get_time(self):
        if self.running:
            return self.elapsed + (time.monotonic() - self.start_time)
        return self.elapsed

# Function to normalize data
def normalize_data(X):
    return (X - X_mean) / (X_std + 1e-8)

# Function to extract landmarks
def extract_landmarks_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, frame

    landmarks = {}
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hand_type = "Left" if results.multi_handedness[idx].classification[0].label == "Left" else "Right"
        landmarks[hand_type] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).flatten()
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return landmarks, frame

# Function to preprocess landmarks
def preprocess_landmarks(landmarks):
    left_hand = landmarks.get('Left', np.zeros(63))
    right_hand = landmarks.get('Right', np.zeros(63))
    return np.concatenate([left_hand, right_hand]).reshape(1, -1)

# Queue for async inference
inference_queue = queue.Queue()

# Background inference thread
def inference_worker():
    while True:
        feature_vector = inference_queue.get()
        if feature_vector is None:
            break

        X_lstm_input = normalize_data(feature_vector).reshape((1, 1, feature_vector.shape[1]))
        predictions = model.predict(X_lstm_input, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        #print(predicted_class)
        inference_queue.task_done()
        inference_results.append(predicted_class)

# Start inference thread
inference_results = deque(maxlen=10)
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# Function to detect gestures using a webcam
def detect_camera():
    #cap = cv2.VideoCapture('data/HandWashDataset/Step_1/HandWash_001_A_01_G01.mp4')
    #cap = cv2.VideoCapture('image_1_test.mp4')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_frame_time = 0
    frame_skip = 2
    frame_counter = 0

    smoothing_threshold = 0.5
    current_step = None
    step_start_time = None
    step_times = {step: 0 for step in step_names}
    last_hand_detected_time = 0

    # Initialize Timer
    timer = ManualTimer()
    hand_detected = False
    wait_start_time = None
    pass_start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, -1)
        frame_counter += 1

        if frame_counter % frame_skip != 0:
            continue

        new_frame_time = time.time()
        fps = int(1 / max(new_frame_time - prev_frame_time, 1e-6))
        prev_frame_time = new_frame_time


        if timer.get_time() > 120:
            if pass_start_time is None:
                pass_start_time = time.monotonic()

            height, width, _ = frame.shape
            cv2.rectangle(frame, (0, height // 3), (width, 2 * height // 3), (0, 255, 0), -1)
            cv2.putText(frame, "Pass", (width // 2 - 50, height // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

            if time.monotonic() - pass_start_time > 5:
                pass_start_time = None
                timer.reset()
                hand_detected = False
                step_start_time = None
                current_step = None
                step_times = {step: 0 for step in step_names}

            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue


        landmarks, frame_with_landmarks = extract_landmarks_from_frame(frame)
        #print(landmarks)

        if landmarks:
            if not hand_detected:
                timer.start()
                hand_detected = True

            feature_vector = preprocess_landmarks(landmarks)
            inference_queue.put(feature_vector)

            if inference_results:
                most_common_prediction = max(set(inference_results), key=inference_results.count)
                confidence = inference_results.count(most_common_prediction) / len(inference_results)
                smoothed_gesture = step_names[most_common_prediction] if confidence >= smoothing_threshold else None

                if smoothed_gesture and smoothed_gesture != current_step:
                    if current_step and step_start_time:
                        step_times[current_step] += timer.get_time() - step_start_time
                    current_step = smoothed_gesture
                    step_start_time = timer.get_time()

            if current_step:
                current_step_time = timer.get_time() - step_start_time
                cv2.putText(frame_with_landmarks, f"Gesture: {current_step}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_with_landmarks, f"Time in {current_step}: {int(step_times[current_step] + current_step_time)}s", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame_with_landmarks, f"Total Time: {int(timer.get_time())}s", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            wait_start_time = None
            last_hand_detected_time = time.time()

        else:
            timer.pause()
            hand_detected = False

            if wait_start_time is None:
                wait_start_time = time.monotonic()

            if time.monotonic() - wait_start_time >= 1 and int(timer.get_time()) > 0:
                short_steps = [
                    step for step, time_spent in step_times.items()
                    if (step in ["1", "3"] and time_spent < 20) or (step not in ["1", "3"] and time_spent < 10)
                ] # 20 sec for both hand
                short_steps_text = " ".join(short_steps)

                height, width, _ = frame_with_landmarks.shape
                cv2.rectangle(frame_with_landmarks, (0, height // 3), (width, 2 * height // 3), (0, 255, 255), -1)

                first_line = "Please follow the step below"
                second_line = "for proper hand hygiene"

                text_size_1 = cv2.getTextSize(first_line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x_1 = (width - text_size_1[0]) // 2
                text_y_1 = height // 2 - 25

                cv2.putText(frame_with_landmarks, first_line, (text_x_1, text_y_1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                text_size_2 = cv2.getTextSize(second_line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x_2 = (width - text_size_2[0]) // 2
                text_y_2 = height // 2 + 15

                cv2.putText(frame_with_landmarks, second_line, (text_x_2, text_y_2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(frame_with_landmarks, f"Steps: {short_steps_text}", (20, height // 2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            if time.time() - last_hand_detected_time > 10:
                hand_detected = False
                wait_start_time = None
                step_start_time = None
                current_step = None
                last_hand_detected_time = 0
                step_times = {step: 0 for step in step_names}
                timer.reset()

            cv2.putText(frame_with_landmarks, "No Hand Detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



        cv2.putText(frame_with_landmarks, f"{fps} FPS", (500, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam Feed", frame_with_landmarks)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        #breakpoint()

    cap.release()
    cv2.destroyAllWindows()
    inference_queue.put(None)
    inference_thread.join()

if __name__ == "__main__":
    detect_camera()
