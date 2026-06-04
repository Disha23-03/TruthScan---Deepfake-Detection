import cv2
import os
import numpy as np
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_REAL_DIR  = "dataset/real"   # put real face images/videos here
RAW_FAKE_DIR  = "dataset/fake"   # put fake/deepfake images/videos here
OUT_DIR       = "processed"
IMG_SIZE      = (224, 224)

# Download from: https://opencv.org/releases/  (no admin needed – just path)
FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      "haarcascade_frontalface_default.xml")

def extract_face(image_path, out_path):
    """Detect face, crop, resize to 224×224, save."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        # No face found – use whole image resized
        face_img = cv2.resize(img, IMG_SIZE)
    else:
        x, y, w, h = faces[0]
        margin = int(0.1 * w)
        x1 = max(0, x - margin); y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        face_img = cv2.resize(img[y1:y2, x1:x2], IMG_SIZE)

    cv2.imwrite(str(out_path), face_img)
    return True

def frames_from_video(video_path, out_folder, label, max_frames=30):
    """Extract up to max_frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    count = 0
    while cap.isOpened() and count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = out_folder / f"{video_path.stem}_f{count}.jpg"
        cv2.imwrite(str(frame_path), cv2.resize(frame, IMG_SIZE))
        count += 1
    cap.release()

def process_directory(src_dir, label_name):
    out_folder = Path(OUT_DIR) / label_name
    out_folder.mkdir(parents=True, exist_ok=True)
    src = Path(src_dir)
    count = 0
    for f in src.glob("*"):
        if f.suffix.lower() in (".mp4", ".avi", ".mov"):
            frames_from_video(f, out_folder, label_name)
            count += 1
        elif f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            out_path = out_folder / f.name
            if extract_face(f, out_path):
                count += 1
    print(f"[{label_name}] Processed {count} files → {out_folder}")

if __name__ == "__main__":
    process_directory(RAW_REAL_DIR, "real")   # label 0
    process_directory(RAW_FAKE_DIR, "fake")   # label 1
    print("✓ Preprocessing complete. Check the 'processed/' folder.")