# import cv2
# import mediapipe as mp
# import numpy as np
# import csv
# import os

# class VideoDataCollector:
#     def __init__(self):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.file_name = 'cricket_shots_data.csv'

#     def process_video_file(self, video_path, class_name):
#         cap = cv2.VideoCapture(video_path)
        
#         # Check if CSV exists, if not, create header
#         if not os.path.exists(self.file_name):
#             self.create_header()

#         print(f"Processing: {video_path} as {class_name}")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # 1. Convert to RGB for MediaPipe
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#             results = self.pose.process(image)
            
#             # 2. Extract Landmarks
#             if results.pose_landmarks:
#                 try:
#                     # Flatten the 33 landmarks into one list of values
#                     pose = results.pose_landmarks.landmark
#                     pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
#                     # Add the label
#                     pose_row.insert(0, class_name)
                    
#                     # Append to CSV
#                     with open(self.file_name, mode='a', newline='') as f:
#                         csv_writer = csv.writer(f)
#                         csv_writer.writerow(pose_row)
                        
#                 except Exception as e:
#                     print(f"Error extracting landmarks: {e}")

#         cap.release()

#     def create_header(self):
#         landmarks = ['class']
#         for val in range(1, 34):
#             landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
#         with open(self.file_name, mode='w', newline='') as f:
#             csv_writer = csv.writer(f)
#             csv_writer.writerow(landmarks)

# # --- USAGE ---
# collector = VideoDataCollector(0)

# collector.process_video_file(path, label)

# print("Data Collection Complete!")









import cv2
import mediapipe as mp
import csv
import os

class LiveDataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.file_name = "cricket_shots_data.csv"
    def collect_from_live(self, class_name):
        cap = cv2.VideoCapture(0)   # <-- webcam
        if not os.path.exists(self.file_name):
            self.create_header()
        print(f"Collecting LIVE data for class: {class_name}")
        print("Press 'q' to stop collection")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                pose = results.pose_landmarks.landmark
                pose_row = []
                for lm in pose:
                    pose_row.extend([lm.x, lm.y, lm.z, lm.visibility])
                pose_row.insert(0, class_name)
                with open(self.file_name, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(pose_row)
            cv2.imshow("Live Pose Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Live Data Collection Complete!")

    def create_header(self):
        header = ['class']
        for i in range(1, 34):
            header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

        with open(self.file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
collector = LiveDataCollector()

label = "Batsman"    
collector.collect_from_live(label)
