import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the T-shirt image (ensure it has a transparent background)
tshirt_image = cv2.imread("tshirt1.png", cv2.IMREAD_UNCHANGED)

# Check if the T-shirt image was loaded properly
if tshirt_image is None or tshirt_image.shape[2] != 4:
    print("Error: T-shirt image not found or does not have an alpha channel.")
    exit()

# Color filters
color_maps = [cv2.COLORMAP_HOT, cv2.COLORMAP_JET, cv2.COLORMAP_OCEAN, cv2.COLORMAP_WINTER]
current_color_index = 0  

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read from webcam.")
        break

    # Flip the frame horizontally for a natural self-view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose detection
    results_pose = pose.process(rgb_frame)

    # Process hand detection
    results_hands = hands.process(rgb_frame)

    # Check for hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_y = wrist.y * h  # Get Y-position of wrist

            # Change color based on hand position (higher = different color)
            if hand_y < h * 0.3:
                current_color_index = 0  # Red (Hot)
            elif hand_y < h * 0.5:
                current_color_index = 1  # Blue (Jet)
            elif hand_y < h * 0.7:
                current_color_index = 2  # Greenish (Ocean)
            else:
                current_color_index = 3  # Light blue (Winter)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        # Calculate width and height based on shoulder distance
        shirt_width = int(abs(right_shoulder.x - left_shoulder.x) * w * 2)
        shirt_height = int((left_hip.y - left_shoulder.y) * h * 2)

        # Ensure the dimensions are within frame bounds
        shirt_width = min(shirt_width, w)
        shirt_height = min(shirt_height, h)

        # Resize only if dimensions are positive
        if shirt_width > 0 and shirt_height > 0:
            tshirt_resized = cv2.resize(tshirt_image, (shirt_width, shirt_height))
        else:
            continue  # Skip frame if resizing is invalid

        # Apply color filter based on hand position
        tshirt_resized[:, :, :3] = cv2.applyColorMap(tshirt_resized[:, :, :3], color_maps[current_color_index])

        # Calculate the center position between shoulders
        center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)  

        # Move the T-shirt slightly upwards to align with the neck
        tshirt_x = max(center_x - (shirt_width // 2), 0)
        tshirt_y = int(neck_y - shirt_height * 0.3)

        # Ensure overlay fits within frame
        shirt_x_end = min(tshirt_x + shirt_width, w)
        shirt_y_end = min(tshirt_y + shirt_height, h)

        # Crop the T-shirt image and alpha mask to fit within frame dimensions
        tshirt_resized = tshirt_resized[:shirt_y_end - tshirt_y, :shirt_x_end - tshirt_x]
        alpha_tshirt = tshirt_resized[:, :, 3] / 255.0  # Normalize alpha channel
        alpha_frame = 1.0 - alpha_tshirt

        # Overlay the T-shirt image on the frame (handling transparency)
        for c in range(3):  
            frame[tshirt_y:shirt_y_end, tshirt_x:shirt_x_end, c] = (
                alpha_tshirt * tshirt_resized[:, :, c] +
                alpha_frame * frame[tshirt_y:shirt_y_end, tshirt_x:shirt_x_end, c]
            )

    # Show the frame
    cv2.putText(frame, "Move Hand Up/Down to Change Color", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Virtual T-Shirt Try-On', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
