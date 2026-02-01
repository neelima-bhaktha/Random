import cv2
import mediapipe as mp

base = cv2.imread("data/normal.jfif")
peace = cv2.imread("data/peace.jfif")
scared = cv2.imread("data/scared.jfif")


if base is None or peace is None or scared is None:
    print("Error: one or more images not found dawg.")
    exit()

current_img = base.copy()

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    face_result = face_mesh.process(rgb_frame)# Run MediaPipe
    hand_result = hands.process(rgb_frame)

    finger_count = 0
    mouth_open = False

    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0].landmark

        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]

        if abs(upper_lip.y - lower_lip.y) > 0.03:
            mouth_open = True

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            finger_count = 0
            for tip, pip in zip(tips, pips):
                if lm[tip].y < lm[pip].y:
                    finger_count += 1

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if mouth_open:
        current_img = scared.copy()
    elif finger_count == 2:
        current_img = peace.copy()
    else:
        current_img = base.copy()

    cv2.imshow("Webcam", frame)
    cv2.imshow("Hamster", current_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
