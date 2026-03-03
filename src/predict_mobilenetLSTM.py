# predict_mobilenetLSTM.py

import os
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
from collections import deque

# IMPORTANT: import custom layer file before loading model
import mobilenet_lstm_model
from mobilenet_lstm_model import build_mobilenet_lstm_model


class MobileNetLSTMLipReader: 
    def __init__(self, model_path, word_labels_path, shape_predictor_path,
                 seq_length=22, img_h=80, img_w=112):

        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded")

        print(f"Loading labels from: {word_labels_path}")
        self.word_labels = np.load(word_labels_path, allow_pickle=True)
        print(f"✅ Loaded {len(self.word_labels)} labels")

        print("Loading dlib detectors...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        print("✅ Face detector loaded")

        # Match training shape
        self.SEQ_LENGTH = seq_length
        self.IMG_H = img_h      # 80
        self.IMG_W = img_w      # 112

        # Buffers
        self.frame_buffer = deque(maxlen=self.SEQ_LENGTH)
        self.prediction_history = deque(maxlen=5)

    # ---------------------------
    # Extract Lip Region
    # ---------------------------
    def extract_lip_region(self, gray_frame, landmarks):
        lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        xs = [p[0] for p in lip_points]
        ys = [p[1] for p in lip_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        pad_w = int(width * 0.2)
        pad_h = int(height * 0.2)

        x_min = max(0, x_min - pad_w)
        x_max = min(gray_frame.shape[1], x_max + pad_w)
        y_min = max(0, y_min - pad_h)
        y_max = min(gray_frame.shape[0], y_max + pad_h)

        lip_region = gray_frame[y_min:y_max, x_min:x_max]
        return lip_region, (x_min, y_min, x_max, y_max)

    # ---------------------------
    # Preprocess Lip Frame
    # ---------------------------
    def preprocess_lip_frame(self, lip_region):
        if lip_region is None or lip_region.size == 0:
            return None

        # Resize to (width=112, height=80)
        lip = cv2.resize(lip_region, (self.IMG_W, self.IMG_H))

        # Gaussian blur
        lip = cv2.GaussianBlur(lip, (5, 5), 0)

        # Contrast stretching (2–98th percentile)
        p2, p98 = np.percentile(lip, (2, 98))
        if p98 - p2 != 0:
            lip = np.clip((lip - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

        # Bilateral filter
        lip = cv2.bilateralFilter(lip, 5, 50, 50)

        # Sharpen
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        lip = cv2.filter2D(lip, -1, kernel)

        # Normalize
        lip = lip.astype(np.float32) / 255.0

        return lip  # shape: (80,112)

    # ---------------------------
    # Predict word
    # ---------------------------
    def predict_word(self):
        if len(self.frame_buffer) < self.SEQ_LENGTH:
            return None, 0.0

        frames = np.array(list(self.frame_buffer), dtype=np.float32)
        frames = np.expand_dims(frames, axis=0)   # (1,22,80,112)
        frames = np.expand_dims(frames, axis=-1)  # (1,22,80,112,1)

        preds = self.model.predict(frames, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        word = self.word_labels[idx]

        # Debug: print every prediction attempt
        print(f"[DEBUG] Model predicted: {word} (conf={conf:.3f})")

        if conf > 0.50:
            return word, conf
        return None, conf

    # ---------------------------
    # Live Loop
    # ---------------------------
    def run_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ERROR: Cannot open webcam.")
            return

        print("\n" + "="*60)
        print("LIVE LIP READING")
        print("="*60)

        frame_count = 0
        current_prediction = None
        current_confidence = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ ERROR: Webcam failed")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.detector(gray, 0)

            # ---------------------------
            # DEBUG PIPELINE
            # ---------------------------
            if len(faces) > 0:
                print("Face detected")

                face = faces[0]
                try:
                    landmarks = self.predictor(gray, face)
                    print("Landmarks OK")
                except:
                    print("❌ Landmark extraction FAILED")
                    self.frame_buffer.clear()
                    continue

                # Lip crop
                lip_region, bbox = self.extract_lip_region(gray, landmarks)
                print("Lip region shape:", None if lip_region is None else lip_region.shape)

                if lip_region is None or lip_region.size == 0:
                    print("❌ Lip region EMPTY")
                    self.frame_buffer.clear()
                    continue

                # Preprocess
                processed = self.preprocess_lip_frame(lip_region)
                print("Processed frame shape:", None if processed is None else processed.shape)

                if processed is None:
                    print("❌ Preprocessing FAILED")
                    self.frame_buffer.clear()
                    continue

                # Add frame
                self.frame_buffer.append(processed)
                print("Buffer length:", len(self.frame_buffer))

                # Predict
                if len(self.frame_buffer) == self.SEQ_LENGTH:
                    pred_word, conf = self.predict_word()
                    print("Pred:", pred_word, "Conf:", conf)

                    if pred_word is not None:
                        current_prediction = pred_word
                        current_confidence = conf

                # Draw lip box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show buffer count
                cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.SEQ_LENGTH}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            else:
                print("No face detected")
                self.frame_buffer.clear()
                current_prediction = None
                cv2.putText(frame, "No face detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            # Draw prediction
            if current_prediction:
                cv2.putText(frame, f"Word: {current_prediction}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {current_confidence:.2%}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            cv2.imshow("MobileNet-LSTM Lip Reader", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Session ended")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    required_files = {
        'model': 'model/best_mobilenet_lstm.h5',
        'labels': 'model/word_labels.npy',
        'predictor': 'model/shape_predictor_68_face_landmarks.dat'
    }

    for key, path in required_files.items():
        if not os.path.exists(path):
            print(f"❌ ERROR: Required file missing: {path}")
            sys.exit(1)

    print("✅ All required files found")

    reader = MobileNetLSTMLipReader(
        model_path=required_files['model'],
        word_labels_path=required_files['labels'],
        shape_predictor_path=required_files['predictor']
    )

    try:
        reader.run_live()
    except Exception as e:
        print("\n❌ ERROR during execution:", e)
        import traceback
        traceback.print_exc()
