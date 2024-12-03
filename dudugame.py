import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

mole_image = cv2.imread("image/dudu.jpg", cv2.IMREAD_UNCHANGED)

mole_width, mole_height = 60, 60

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mole_x, mole_y = random.randint(0, frame_width - mole_width), random.randint(0, frame_height - mole_height)

last_mole_change_time = time.time()
mole_hidden_time = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mole_resized = cv2.resize(mole_image, (mole_width, mole_height))

    if mole_resized.shape[2] == 4:
        bgr = mole_resized[:, :, :3]
        alpha = mole_resized[:, :, 3]
        alpha = alpha / 255.0
       
        mole_x = min(mole_x, frame_width - mole_width)
        mole_y = min(mole_y, frame_height - mole_height)

        roi = frame[mole_y:mole_y + mole_height, mole_x:mole_x + mole_width]

        if roi.shape[:2] == alpha.shape:  
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + bgr[:, :, c] * alpha
            
            frame[mole_y:mole_y + mole_height, mole_x:mole_x + mole_width] = roi
    else:
        frame[mole_y:mole_y + mole_height, mole_x:mole_x + mole_width] = mole_resized

    if time.time() - mole_hidden_time > 2:  
        if time.time() - last_mole_change_time > 2:
            mole_x, mole_y = random.randint(0, frame_width - mole_width), random.randint(0, frame_height - mole_height)
            last_mole_change_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if abs(cx - (mole_x + mole_width // 2)) < mole_width // 2 and abs(cy - (mole_y + mole_height // 2)) < mole_height // 2:
                    print("Mole hit!")
                    mole_hidden_time = time.time()  
                    mole_x, mole_y = random.randint(0, frame_width - mole_width), random.randint(0, frame_height - mole_height)

    cv2.imshow("Dudu Game", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()