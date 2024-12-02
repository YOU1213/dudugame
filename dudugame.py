import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

mole_image = cv2.imread("image/dudu.jpg", cv2.IMREAD_UNCHANGED)

mole_x, mole_y = random.randint(50, 500), random.randint(50, 500)
last_mole_change_time = time.time()
mole_hidden_time = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mole_resized = cv2.resize(mole_image, (60, 60))

    if mole_resized.shape[2] == 4:
        bgr = mole_resized[:, :, :3]
        alpha = mole_resized[:, :, 3]
        alpha = alpha / 255.0
        
        roi = frame[mole_y:mole_y + mole_resized.shape[0], mole_x:mole_x + mole_resized.shape[1]]

        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + bgr[:, :, c] * alpha
        
        frame[mole_y:mole_y + mole_resized.shape[0], mole_x:mole_x + mole_resized.shape[1]] = roi
    else:
        frame[mole_y:mole_y + mole_resized.shape[0], mole_x:mole_x + mole_resized.shape[1]] = mole_resized

    if time.time() - mole_hidden_time > 2:  
        if time.time() - last_mole_change_time > 2:
            mole_x, mole_y = random.randint(50, 500), random.randint(50, 500)
            last_mole_change_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if abs(cx - mole_x) < 30 and abs(cy - mole_y) < 30:
                    print("Mole hit!")
                    mole_hidden_time = time.time()  
                    mole_x, mole_y = random.randint(50, 500), random.randint(50, 500)  # 두더지 위치 재설정

    cv2.imshow("Dudu Game", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
