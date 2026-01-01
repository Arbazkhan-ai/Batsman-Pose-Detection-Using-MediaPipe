import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

# Load trained model
with open('cricket_shot_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # If pose detected → predict
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                row = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks
                ]).flatten()

                X = pd.DataFrame([row])

                shot = model.predict(X)[0]
                prob = model.predict_proba(X)[0].max()

                # Display result only (no landmarks)
                cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)
                cv2.putText(
                    frame, f'SHOT: {shot}',
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f'CONF: {round(prob, 2)}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1
                )

            except Exception:
                pass

        cv2.imshow('Cricket Shot Detector', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
