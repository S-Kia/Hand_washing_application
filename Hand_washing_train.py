import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Mapping of step names
step_names = ["1", "2lt", "2rt", "3", "4lt", "4rt", "5lt", "5rt", "6lt", "6rt", "7lt", "7rt"]

# Function to parse hand data from a string
def parse_hand_data(hand_str):
    try:
        hand_data = np.array([float(v) for v in hand_str.strip('[]').split(',')])
        return hand_data
    except Exception:
        return np.array([0.0] * 21 * 3)

# Function to load data
def load_data(data_folder):
    X, y = [], []
    for step_idx, step_name in enumerate(step_names):
        file_path = os.path.join(data_folder, f"{step_name}.csv")
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)

            # Parse left and right hand data
            left_data = df['Left'].apply(parse_hand_data)
            right_data = df['Right'].apply(parse_hand_data)
            step_data = np.concatenate([np.stack(left_data), np.stack(right_data)], axis=1)

            # Append data and labels
            X.append(step_data)
            y.extend([step_idx] * len(step_data))  # Label corresponds to the step index

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y

# Function to normalize the data
def normalize_data(all_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data)
    return scaled_data

# Function to create an LSTM model
def create_lstm_model(input_shape, num_classes, dropout):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(dropout),
        LSTM(128, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to visualize the distribution of steps (labels)
def visualize_step_distribution(y, step_labels):
    step_counts = [np.sum(y == step) for step in range(len(step_labels))]
    print(step_counts)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=step_labels, y=step_counts, palette='viridis')
    plt.title("Step Distribution")
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.show()

# Function to visualize accuracy over epochs
def visualize_accuracy_and_epochs(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to visualize hand keypoint distribution for all steps
def visualize_hand_distribution_all_steps(X, y, num_steps=12):
    # Visualize the distribution of hand keypoints for each step
    for step in range(num_steps):
        # Extract step-specific data
        step_data = X[y == step]
        num_features = step_data.shape[1] // 2  # Left and right hand data are concatenated
        left_hand = step_data[:, :num_features].reshape(-1, 21, 3)
        right_hand = step_data[:, num_features:].reshape(-1, 21, 3)

        # Filter out invalid keypoints with coordinates (0, 0, 0)
        left_hand = left_hand[np.any(left_hand != (0, 0, 0), axis=2)]
        right_hand = right_hand[np.any(right_hand != (0, 0, 0), axis=2)]

        # Reshape again after filtering
        left_hand = left_hand.reshape(-1, 21, 3)
        right_hand = right_hand.reshape(-1, 21, 3)

        # Count total keypoints for each hand
        left_hand_points = left_hand.shape[0] * left_hand.shape[1]
        right_hand_points = right_hand.shape[0] * right_hand.shape[1]

        # Plot the hand keypoints
        plt.figure(figsize=(10, 6))
        for frame in left_hand:
            plt.scatter(frame[:, 0], frame[:, 1], color='blue', alpha=0.7)
        for frame in right_hand:
            plt.scatter(frame[:, 0], frame[:, 1], color='red', alpha=0.7)

        # Add Legend with keypoint counts
        plt.scatter([], [], color='blue', label=f'Left Hand ({left_hand_points} points)')
        plt.scatter([], [], color='red', label=f'Right Hand ({right_hand_points} points)')

        plt.title(f"Hand Distribution (Step {step_names[step]})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()

# Main pipeline for gesture recognition
def generalize_recognition(data_folder):
    X, y = load_data(data_folder)

    print(X.shape, y.shape)

    # Save the dataset for normalization
    joblib.dump(X, 'X.pkl')

    # Visualize the step distribution
    visualize_step_distribution(y, step_names)

    # Visualize the hand distribution for all steps
    visualize_hand_distribution_all_steps(X, y, num_steps=len(step_names))

    # Normalize the data
    X_normalized = normalize_data(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Reshape the data for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create and train the LSTM model
    model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), len(step_names), 0.2)
    model.summary()
    history = model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test))

    # Visualize accuracy
    visualize_accuracy_and_epochs(history)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test_lstm), axis=1)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2%}")

    # Save the trained model
    model.save('generalized_lstm_model.keras')

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=step_names, yticklabels=step_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_and_plot_metrics_vs_dropout(data_folder):
    X, y = load_data(data_folder)
    X_normalized = normalize_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    dropout_values = [round(0.1 * i, 1) for i in range(1, 10)]
    accuracy_list, f1_list, recall_list = [], [], []

    for dropout in dropout_values:
        print(f"\nTraining with dropout = {dropout}")
        model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), len(step_names), dropout)
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)  # reduce epochs for speed
        y_pred = np.argmax(model.predict(X_test_lstm), axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        accuracy_list.append(acc)
        f1_list.append(f1)
        recall_list.append(rec)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=dropout_values, y=accuracy_list, marker='o', label='Accuracy')
    sns.lineplot(x=dropout_values, y=f1_list, marker='o', label='F1 Score')
    sns.lineplot(x=dropout_values, y=recall_list, marker='o', label='Recall')
    plt.title("Metrics vs Dropout")
    plt.xlabel("Dropout Value")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generalize_recognition_timestep(data_folder, time_steps):
    # Function to reshape data into sequences of fixed time steps
    def reshape_to_sequences(X, y, time_steps):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_sequences = num_samples // time_steps

        X_seq = X[:num_sequences * time_steps].reshape((num_sequences, time_steps, num_features))
        y_seq = y[:num_sequences * time_steps].reshape((num_sequences, time_steps))

        # Use majority label in each sequence as the sequence label
        y_seq = [np.bincount(seq).argmax() for seq in y_seq]
        return X_seq, np.array(y_seq)

    # Load data
    X, y = load_data(data_folder)
    print("Original shape:", X.shape, y.shape)

    # Save raw dataset for inspection
    joblib.dump(X, 'X.pkl')

    # Normalize the data
    X_normalized = normalize_data(X)

    # Reshape into sequences of given timesteps
    X_seq, y_seq = reshape_to_sequences(X_normalized, y, time_steps=time_steps)
    print("Reshaped to sequences:", X_seq.shape, y_seq.shape)

    # Split into training and testing sets
    X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Create and train the LSTM model
    model = create_lstm_model((time_steps, X_train_lstm.shape[2]), len(step_names), 0.2)
    model.summary()
    history = model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test))

    # Visualize accuracy
    visualize_accuracy_and_epochs(history)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test_lstm), axis=1)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2%}")

    # Save the trained model (optional)
    # model.save(f'generalized_lstm_model_{time_steps}steps.keras')

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=step_names, yticklabels=step_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_metrics_by_timestep(data_folder, max_steps=10):
    accuracy_list = []
    f1_list = []
    recall_list = []
    time_steps_range = list(range(1, max_steps + 1))

    # Load and normalize once to save time
    X, y = load_data(data_folder)
    X_normalized = normalize_data(X)

    for time_steps in time_steps_range:
        print(f"\nEvaluating for time_steps = {time_steps}")

        # Skip if not enough data for the timestep
        if X_normalized.shape[0] < time_steps:
            accuracy_list.append(0)
            f1_list.append(0)
            recall_list.append(0)
            continue

        # Reshape into sequences
        def reshape_to_sequences(X, y, time_steps):
            num_samples = X.shape[0]
            num_features = X.shape[1]
            num_sequences = num_samples // time_steps

            X_seq = X[:num_sequences * time_steps].reshape((num_sequences, time_steps, num_features))
            y_seq = y[:num_sequences * time_steps].reshape((num_sequences, time_steps))
            y_seq = [np.bincount(seq).argmax() for seq in y_seq]
            return X_seq, np.array(y_seq)

        X_seq, y_seq = reshape_to_sequences(X_normalized, y, time_steps)

        if len(X_seq) == 0:
            accuracy_list.append(0)
            f1_list.append(0)
            recall_list.append(0)
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        model = create_lstm_model((time_steps, X_train.shape[2]), len(step_names), 0.2)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # fewer epochs for speed
        y_pred = np.argmax(model.predict(X_test), axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        accuracy_list.append(acc)
        f1_list.append(f1)
        recall_list.append(rec)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=time_steps_range, y=accuracy_list, marker='o', label='Accuracy')
    sns.lineplot(x=time_steps_range, y=f1_list, marker='o', label='F1 Score')
    sns.lineplot(x=time_steps_range, y=recall_list, marker='o', label='Recall')
    plt.title("Metrics vs Time Step")
    plt.xlabel("Time Steps")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Execute the main pipeline
if __name__ == "__main__":
    data_folder = 'data/landmark'  # Path to the folder containing the CSV files (e.g., 1.csv, 2lt.csv, etc.)
    #evaluate_and_plot_metrics_vs_dropout(data_folder)
    #generalize_recognition_timestep(data_folder, 5)
    #evaluate_metrics_by_timestep(data_folder, max_steps=30)
    generalize_recognition(data_folder)


