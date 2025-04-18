import cv2
import numpy as np
import mediapipe as mp
import autopy
import pyautogui
import util
import random
from pynput.mouse import Button, Controller

# Constants
screen_width, screen_height = autopy.screen.size()
print(screen_width,screen_height)
frame_width, frame_height = 840, 680
frame_margin = 100
smoothening = 8  # Higher values = smoother but slower response

# Mouse controller
mouse = Controller()

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1  # Focus on single hand for better performance
)

# State tracking
prev_x, prev_y = 0,0                      #screen_width // 2, screen_height // 2

# Function to find the tip of the index finger from the processed hand landmarks
def find_finger_tip(processed):
    # Check if any hand landmarks were detected
    if processed.multi_hand_landmarks:
        # Get landmarks of the first detected hand (assuming one hand is present)
        hand_landmarks = processed.multi_hand_landmarks[0]

        # Extract the landmark point for the index finger tip
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Return the position of the index finger tip
        return index_finger_tip

    # If no hand detected, return None values
    return None

def move_mouse(index_finger_tip):
    """Convert finger position to smooth mouse movement"""
    # Declare global variables to store previous mouse pointer position
    global prev_x, prev_y
    
    # If no valid index finger tip position is provided, return without doing anything
    if index_finger_tip is None:
        return
        
    # Calculate dimensions of the active movement area by subtracting margins
    # Active area is smaller than the full frame to ignore movements near the edges
    active_width = frame_width - 2 * frame_margin
    active_height = frame_height - 2 * frame_margin
    # Adjust finger tip position relative to the active movement area
    # Normalize coordinates to account for the margins and map them to a range [0, 1]
    x = (index_finger_tip.x * frame_width - frame_margin) / active_width
    y = (index_finger_tip.y * frame_height - frame_margin) / active_height
    # Clamp the normalized coordinates to ensure they stay within [0, 1]
    # Prevents the coordinates from exceeding the boundaries of the active region
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    # Map the normalized coordinates to screen dimensions (convert to pixel positions)
    # screen_x and screen_y represent the target mouse pointer positions on the screen
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    # Smooth the movement by interpolating between previous and target positions
    # Reduces abrupt jumps in pointer movement for a more natural feel
    curr_x = prev_x + (screen_x - prev_x) // smoothening
    curr_y = prev_y + (screen_y - prev_y) // smoothening
    # Clamp the final pointer positions to ensure they stay within the screen boundaries
    curr_x = np.clip(curr_x, 0, screen_width - 1)
    curr_y = np.clip(curr_y, 0, screen_height - 1)
    # Move the mouse pointer to the calculated position using autopy
    autopy.mouse.move(curr_x, curr_y)
    # Update previous position variables for continuity in smoothing during the next call
    prev_x, prev_y = curr_x, curr_y


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )
def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )

def is_scroll_up(landmark_list):
    """Detect scroll up gesture (index finger moves up vertically)"""
    return (landmark_list[8][1] < landmark_list[6][1])  # Index finger tip is higher than its second joint


def is_scroll_down(landmark_list):
    """Detect scroll down gesture (index finger moves down vertically)"""
    return (landmark_list[8][1] > landmark_list[6][1])  # Index finger tip is lower than its second joint
    

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50  and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        elif is_scroll_up(landmark_list):
            pyautogui.scroll(30)  # Positive value scrolls up
            cv2.putText(frame, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        # Scroll down gesture
        elif is_scroll_down(landmark_list):
            pyautogui.scroll(-30)  # Negative value scrolls down
            cv2.putText(frame, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

def main():
    """Main program loop"""
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Frame processing
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (20, 50), (820, 660), (255, 0, 255), 2)
            
            # Convert to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frame_rgb)
            
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))
            
            detect_gesture(frame, landmark_list, processed)
            
            cv2.imshow("Virtual Mouse", frame)
            if cv2.waitKey(1) == ord('q'):
                break           
    finally:
        cap.release()
        cv2.destroyAllWindows()

main()
