import cv2
import dlib
import os
import time
import json
from datetime import datetime

# ==== CONFIGURATION - SET THESE BEFORE RUNNING ====
WORD = "help"  # Change this for each word
SPEAKER_ID = "me"  # Single speaker mode
LIGHTING = "auto"  # Auto-detect or set to: "bright", "dim", "normal", "backlit"
ANGLE = "auto"  # Auto-set or: "front", "left", "right", "up", "down"
DISTANCE = "auto"  # Auto-set or: "close", "medium", "far"
NOTES = ""  # Any additional notes

# AUTO-VARIATION MODE: Automatically cycles through conditions
AUTO_VARIATION = True  # Set to True for automatic condition cycling

# ==== PATHS ====
OUTPUT_DIR = "data/"
METADATA_FILE = "data/metadata.json"

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Constants
FRAMES_PER_WORD = 22

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load or create metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"takes": [], "speakers": {}}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def add_take_to_metadata(word, take_num, speaker, lighting, angle, distance, notes):
    metadata = load_metadata()
    
    take_info = {
        "word": word,
        "take_number": take_num,
        "speaker_id": speaker,
        "lighting": lighting,
        "angle": angle,
        "distance": distance,
        "timestamp": datetime.now().isoformat(),
        "path": f"data/{word}/take_{take_num}",
        "frames": FRAMES_PER_WORD,
        "resolution": "112x80",
        "notes": notes
    }
    
    metadata["takes"].append(take_info)
    save_metadata(metadata)
    print(f"✅ Metadata logged for take {take_num}")

# Start capturing video
cap = cv2.VideoCapture(0)

print("\n" + "="*60)
print(f"📹 RECORDING SETUP")
print("="*60)
print(f"Word:     {WORD}")
print(f"Speaker:  {SPEAKER_ID}")
print(f"Lighting: {LIGHTING}")
print(f"Angle:    {ANGLE}")
print(f"Distance: {DISTANCE}")
print(f"Notes:    {NOTES if NOTES else 'None'}")
print("="*60)
print("\n⌨️  Press 'L' to start recording")
print("⌨️  Press 'Q' to quit")
print("⌨️  Press 'C' to change settings\n")

recording = False
frame_count = 0
take_number = 1

# Find the next available take number
word_dir = os.path.join(OUTPUT_DIR, WORD)
if not os.path.exists(word_dir):
    os.makedirs(word_dir)

while os.path.exists(os.path.join(word_dir, f"take_{take_number}")):
    take_number += 1

# Track total takes for this session
session_takes = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Display info on screen
    info_y = 30
    cv2.putText(frame, f"Word: {WORD}", (10, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Take: {take_number}", (10, info_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Session: {session_takes} takes", (10, info_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if recording:
        cv2.putText(frame, f"RECORDING: {frame_count}/{FRAMES_PER_WORD}", 
                   (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract lip region coordinates
        x_min = min([landmarks.part(i).x for i in range(48, 68)])
        x_max = max([landmarks.part(i).x for i in range(48, 68)])
        y_min = min([landmarks.part(i).y for i in range(48, 68)])
        y_max = max([landmarks.part(i).y for i in range(48, 68)])
        
        # Expand bounding box
        EXPAND_RATIO = 1.3
        lip_width = x_max - x_min
        lip_height = y_max - y_min

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        x_min = max(0, int(x_center - (lip_width // 2) * EXPAND_RATIO))
        x_max = min(frame.shape[1], int(x_center + (lip_width // 2) * EXPAND_RATIO))
        y_min = max(0, int(y_center - (lip_height // 2) * EXPAND_RATIO))
        y_max = min(frame.shape[0], int(y_center + (lip_height // 2) * EXPAND_RATIO))

        # Draw rectangle around lips
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # If recording, save frames
        if recording and frame_count < FRAMES_PER_WORD:
            lip_region = frame[y_min:y_max, x_min:x_max]
            lip_region = cv2.resize(lip_region, (112, 80))

            # Create the take folder
            take_dir = os.path.join(word_dir, f"take_{take_number}")
            if not os.path.exists(take_dir):
                os.makedirs(take_dir)

            # Save the frame
            frame_path = os.path.join(take_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, lip_region)
            frame_count += 1

            # Stop recording after collecting enough frames
            if frame_count >= FRAMES_PER_WORD:
                print(f"\n✅ Recorded {FRAMES_PER_WORD} frames for '{WORD}' (take_{take_number})")
                
                # Log to metadata
                add_take_to_metadata(WORD, take_number, SPEAKER_ID, 
                                   LIGHTING, ANGLE, DISTANCE, NOTES)
                
                recording = False
                frame_count = 0
                session_takes += 1
                take_number += 1
                
                print(f"📊 Session total: {session_takes} takes")
                print("🎬 Ready for next take! Press 'L' when ready.\n")

    # Display webcam feed
    cv2.imshow(f"Lip Reader - Recording '{WORD}'", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l') and not recording:
        print(f"\n🔴 Recording '{WORD}' (take {take_number})... Speak now!")
        recording = True
        frame_count = 0
    elif key == ord('c'):
        print("\n⚙️  To change settings, edit the variables at the top of the script:")
        print(f"   WORD = '{WORD}'")
        print(f"   SPEAKER_ID = '{SPEAKER_ID}'")
        print(f"   LIGHTING = '{LIGHTING}'")
        print(f"   ANGLE = '{ANGLE}'")
        print(f"   DISTANCE = '{DISTANCE}'")
        print()

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Session complete! Recorded {session_takes} takes.")
print(f"📁 Data saved in: {word_dir}")
print(f"📋 Metadata saved in: {METADATA_FILE}")