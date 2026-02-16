import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration du détecteur de mains
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.5,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# Connexions des points de la main (indices des landmarks à connecter)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Majeur
    (9, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),  # Auriculaire
    (0, 17)  # Paume
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Dessiner les points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Dessiner les connexions
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

            # Calculer la distance entre le pouce et l'index
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            dist = math.hypot(thumb.x - index.x, thumb.y - index.y)

            if dist < 0.03:
                cv2.putText(frame, "PINCH (zoom)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif dist > 0.08:
                cv2.putText(frame, "OPEN (dezoom)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if (key == 27):
        break

cap.release()
cv2.destroyAllWindows()
