import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE=False
# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

# Screen size
screen_w, screen_h = pyautogui.size()

# Move mouse to center of the screen at start
pyautogui.moveTo(screen_w // 2, screen_h // 2)

# Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Thresholds
BLINK_THRESHOLD = 5.0
DOUBLE_BLINK_TIME = 0.5
LONG_BLINK_DURATION = 1.0  # Seconds

# Track blink and head position
last_blink_time = 0
blink_count = 0
blink_start_time = None
initial_nose_y = None
initial_nose_z = None
initial_nose_x = None

# Toggle flag for controlling movement
mouse_control_enabled = True

# Live action label
last_action = ""
last_action_time = 0
action_display_duration = 2  # seconds

# Movement sensitivity (reduce multiplier to slow down movement)
SENSITIVITY = 0.6  # Reduce this to slow down

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    action = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        frame_h, frame_w, _ = frame.shape

        # Draw face mesh landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

        # Eye coordinates
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[374], landmarks[386]]

        left_eye_ratio = euclidean_distance(
            (left_eye[0].x * frame_w, left_eye[0].y * frame_h),
            (left_eye[1].x * frame_w, left_eye[1].y * frame_h)
        )
        right_eye_ratio = euclidean_distance(
            (right_eye[0].x * frame_w, right_eye[0].y * frame_h),
            (right_eye[1].x * frame_w, right_eye[1].y * frame_h)
        )

        both_eyes_closed = left_eye_ratio < BLINK_THRESHOLD and right_eye_ratio < BLINK_THRESHOLD

        # Long blink detection for pause/resume
        if both_eyes_closed:
            if blink_start_time is None:
                blink_start_time = current_time
            elif current_time - blink_start_time >= LONG_BLINK_DURATION:
                mouse_control_enabled = not mouse_control_enabled
                action = "Mouse Paused" if not mouse_control_enabled else "Mouse Resumed"
                blink_start_time = None
                time.sleep(1.0)  # debounce
        else:
            blink_start_time = None

        # Short blink detection
        if left_eye_ratio < BLINK_THRESHOLD and right_eye_ratio >= BLINK_THRESHOLD:
            pyautogui.click(button='left')
            action = "Left Click"
            time.sleep(0.2)
        elif right_eye_ratio < BLINK_THRESHOLD and left_eye_ratio >= BLINK_THRESHOLD:
            pyautogui.click(button='right')
            action = "Right Click"
            time.sleep(0.2)
        elif both_eyes_closed:
            if current_time - last_blink_time < DOUBLE_BLINK_TIME:
                blink_count += 1
            else:
                blink_count = 1
            last_blink_time = current_time
            if blink_count == 2:
                pyautogui.doubleClick()
                action = "Double Click"
                blink_count = 0
            time.sleep(0.2)

        # Nose tip for head movement
        nose_tip = landmarks[1]
        nose_x = nose_tip.x
        nose_y = nose_tip.y
        nose_z = nose_tip.z

        if initial_nose_y is None:
            initial_nose_y = nose_y
        if initial_nose_z is None:
            initial_nose_z = nose_z
        if initial_nose_x is None:
            initial_nose_x = nose_x

        # Scroll
        if nose_y < initial_nose_y - 0.02:
            pyautogui.scroll(50)
            action = "Scroll Up"
        elif nose_y > initial_nose_y + 0.02:
            pyautogui.scroll(-50)
            action = "Scroll Down"

        # Zoom
        if nose_z < initial_nose_z - 0.02:
            pyautogui.hotkey('ctrl', '-')
            action = "Zoom Out"
            time.sleep(0.2)
        elif nose_z > initial_nose_z + 0.02:
            pyautogui.hotkey('ctrl', '+')
            action = "Zoom In"
            time.sleep(0.2)

        # Real-time head-based mouse movement
        if mouse_control_enabled:
            dx = (nose_x - initial_nose_x) * screen_w * SENSITIVITY
            dy = (nose_y - initial_nose_y) * screen_h * SENSITIVITY
            current_mouse_x, current_mouse_y = pyautogui.position()
            new_mouse_x = np.clip(current_mouse_x + dx, 0, screen_w - 1)
            new_mouse_y = np.clip(current_mouse_y + dy, 0, screen_h - 1)
            pyautogui.moveTo(new_mouse_x, new_mouse_y)

    # Show recent action on screen
    if action:
        last_action = action
        last_action_time = current_time
    if current_time - last_action_time < action_display_duration and last_action:
        cv2.putText(frame, last_action, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Indicate mouse control status
    status_text = "Mouse Control: ON" if mouse_control_enabled else "Mouse Control: OFF"
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
