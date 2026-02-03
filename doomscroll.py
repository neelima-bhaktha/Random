import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os


class DoomscrollDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.roasts = [
            "You'll fail if you don't stop!",
            "Your dreams called - they want your attention back!",
            "Scrolling won't make that deadline disappear!",
            "The phone can wait. Your future can't.",
            "Success doesn't scroll itself into existence!",
            "That screen won't study for you!",
            "Your goals > Your feed. Remember that.",
            "Future you is watching. They're disappointed.",
            "Every scroll is a step backward. Look up!",
            "The algorithm wins again. Pathetic.",
            "Is this really more important than your goals?",
            "Your productivity just left the chat.",
            "DOOMSCROLLING DETECTED!",
            "PUT. THE. PHONE. DOWN. NOW.",
            "This is why you're behind schedule."
        ]

        self.last_roast_time = 0
        self.roast_cooldown = 3
        self.current_roast = ""

        self.rickroll_path = "RickRoll.mp4"
        self.rickroll_cap = None
        self.is_rickrolling = False

        self.doomscroll_count = 0
        self.normal_count = 0
        self.detection_threshold = 3

    def detect_doomscroll_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark
        ys = np.array([lm.y for lm in landmarks])

        forehead_y = np.min(ys)
        chin_y = np.max(ys)
        face_height = chin_y - forehead_y

        if face_height < 1e-6:
            return False

        nose_y = landmarks[1].y
        nose_ratio = (nose_y - forehead_y) / face_height

        return nose_ratio > 0.6

    def play_rickroll(self):
        if not self.is_rickrolling and os.path.exists(self.rickroll_path):
            self.rickroll_cap = cv2.VideoCapture(self.rickroll_path)
            self.is_rickrolling = True

    def stop_rickroll(self):
        if self.is_rickrolling:
            self.is_rickrolling = False
            if self.rickroll_cap:
                self.rickroll_cap.release()
            self.rickroll_cap = None
            cv2.destroyWindow("Rickroll")

    def show_roast(self, frame):
        current_time = time.time()

        if current_time - self.last_roast_time > self.roast_cooldown:
            self.current_roast = random.choice(self.roasts)
            self.last_roast_time = current_time

        overlay = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "DOOMSCROLLING DETECTED!", (w // 2 - 260, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

        cv2.putText(frame, self.current_roast, (w // 2 - 350, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            raw_detection = self.detect_doomscroll_mediapipe(frame)

            if raw_detection:
                self.doomscroll_count += 1
                self.normal_count = 0
            else:
                self.normal_count += 1
                self.doomscroll_count = 0

            is_doomscrolling = self.doomscroll_count >= self.detection_threshold
            is_normal = self.normal_count >= self.detection_threshold

            if is_doomscrolling:
                self.show_roast(frame)
                self.play_rickroll()
            elif is_normal:
                cv2.putText(frame, "Good posture! Keep it up!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.stop_rickroll()

            if self.is_rickrolling and self.rickroll_cap:
                ret, video_frame = self.rickroll_cap.read()
                if not ret:
                    self.rickroll_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    video_frame = cv2.resize(video_frame, (320, 180))
                    cv2.imshow("Rickroll", video_frame)

            cv2.imshow("Doomscrolling Blocker", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.stop_rickroll()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DoomscrollDetector().run()
