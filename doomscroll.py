import cv2
import mediapipe as mp
import numpy as np
import random
import time
import threading
import subprocess
import os


class Doomscroll:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
)
        
        self.roast = [
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
            "Doomscrolling detected! You're better than this!",
            "PUT. THE. PHONE. DOWN. NOW.",
            "This is why you're behind schedule."
        ]


        