# realtime_predict.py
import cv2, json, joblib, numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

MODEL_PATH = "saved_models/lstm_cricket.h5"   # or whichever you prefer
SCALER_PATH = "saved_models/scaler.save"
LABEL_MAP = "saved_models/label_map.json"
SEQ_LEN = 30

# load
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(LABEL_MAP, "r") as f:
    classes = json.load(f)["classes"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

buffer = []   # stores last SEQ_LEN frame features

cap = cv2.VideoCapture(0)  # change to path for a video file
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    res = pose.process(img_rgb)
    img_rgb.flags.writeable = True

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        feat = []
        for p in lm:
            feat.extend([p.x, p.y, p.z, p.visibility])
        buffer.append(feat)
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        if len(buffer) == SEQ_LEN:
            X = np.array(buffer, dtype=np.float32).reshape(-1, len(feat))
            Xs = scaler.transform(X).reshape(1, SEQ_LEN, -1)
            preds = model.predict(Xs)
            pred_idx = preds.argmax(axis=1)[0]
            label = classes[pred_idx]
            conf = preds[0][pred_idx]
            cv2.putText(frame, f"{label} ({conf:.2f})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
