import cv2
import mediapipe as mp
import time
import math

# -------------------- MediaPipe Setup --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# -------------------- Landmark Indexes --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

# -------------------- Blink Parameters --------------------
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3

eye_closed_frames = 0
blink_count = 0
blink_timestamps = []

# -------------------- Yawn Parameters --------------------
MAR_THRESHOLD = 0.6
YAWN_DURATION = 1.0  # seconds

yawn_count = 0
mouth_open_start = None

# -------------------- Utility Functions --------------------
def euclidean(p1, p2):
    return math.dist(p1, p2)

def calculate_ear(landmarks, eye, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    return (euclidean(pts[1], pts[5]) + euclidean(pts[2], pts[4])) / (2 * euclidean(pts[0], pts[3]))

def calculate_mar(landmarks, mouth, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth]
    vertical = euclidean(pts[1], pts[7]) + euclidean(pts[2], pts[6]) + euclidean(pts[3], pts[5])
    horizontal = euclidean(pts[0], pts[4])
    return vertical / (2 * horizontal)

def detect_blink(ear):
    global eye_closed_frames, blink_count, blink_timestamps
    if ear < EAR_THRESHOLD:
        eye_closed_frames += 1
    else:
        if eye_closed_frames >= CONSEC_FRAMES:
            blink_count += 1
            blink_timestamps.append(time.time())
        eye_closed_frames = 0

def detect_yawn(mar):
    global mouth_open_start, yawn_count
    if mar > MAR_THRESHOLD:
        if mouth_open_start is None:
            mouth_open_start = time.time()
        elif time.time() - mouth_open_start > YAWN_DURATION:
            yawn_count += 1
            mouth_open_start = None
    else:
        mouth_open_start = None

def blink_rate():
    return len([t for t in blink_timestamps if t > time.time() - 60])

# -------------------- Webcam Loop --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            ear = (calculate_ear(face.landmark, LEFT_EYE, w, h) +
                   calculate_ear(face.landmark, RIGHT_EYE, w, h)) / 2

            mar = calculate_mar(face.landmark, MOUTH, w, h)

            detect_blink(ear)
            detect_yawn(mar)

            mp_drawing.draw_landmarks(
                frame, face, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )

    # -------------------- UI --------------------
    cv2.putText(frame, f"Blinks/min: {blink_rate()}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Yawns: {yawn_count}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Airport Behavioral Risk – Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
