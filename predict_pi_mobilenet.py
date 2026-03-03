import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="best_mobilenet_lstm_quantized.tflite")
parser.add_argument("--labels", default="word_labels.npy")
parser.add_argument("--stream", default="")
parser.add_argument("--seq_len", type=int, default=22)
parser.add_argument("--w", type=int, default=112)
parser.add_argument("--h", type=int, default=80)
args = parser.parse_args()

# Load model
print("Loading TFLite model...")
interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")

# Load word labels
labels = np.load(args.labels, allow_pickle=True)
print("Labels:", labels)

# Camera
if args.stream:
    print("Using iPhone camera stream:", args.stream)
    cap = cv2.VideoCapture(args.stream)
else:
    print("Using USB webcam")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("ERROR: Could not open camera/stream.")

# Simple face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

seq = []
recording = False

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (args.w, args.h))
    gray = gray.astype(np.float32) / 255.0
    return gray

print("\nPress L to record, Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    disp = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Estimate mouth region = bottom 1/3 of face
        y1 = y + int(0.60 * h)
        y2 = y + int(0.95 * h)
        x1 = x + int(0.15 * w)
        x2 = x + int(0.85 * w)

        mouth = frame[y1:y2, x1:x2]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if recording:
            if len(seq) < args.seq_len and mouth.size > 0:
                seq.append(preprocess(mouth))
                cv2.putText(disp, f"Recording {len(seq)}/{args.seq_len}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if len(seq) == args.seq_len:
                # Prepare input
                inp = np.array(seq, dtype=np.float32)
                inp = inp.reshape(1, args.seq_len, args.h, args.w, 1)

                interpreter.set_tensor(input_details[0]['index'], inp)
                t0 = time.time()
                interpreter.invoke()
                t1 = time.time()

                out = interpreter.get_tensor(output_details[0]['index'])[0]
                idx = np.argmax(out)
                conf = out[idx]

                print(f"\nPrediction: {labels[idx]}  Conf={conf:.3f}  Time={t1-t0:.3f}s\n")

                seq = []
                recording = False

    cv2.imshow("LipReader SAFE MODE", disp)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    if k == ord('l') and not recording:
        print("\nRecording started. Please mouth the word...")
        recording = True
        seq = []

cap.release()
cv2.destroyAllWindows()
