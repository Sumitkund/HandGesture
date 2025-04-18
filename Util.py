import numpy as np

def get_angle(a,b,c):
    # Step 1: Calculate the angles of vectors b→c and b→a relative to the x-axis
    angle_bc=np.arctan2(c[1] - b[1],c[0] - b[0])  # Angle of vector b→c
    angle_ba=np.arctan2(a[1] - b[1],a[0] - b[0])  # Angle of vector b→a
    # Step 2: Find the difference between these angles
    radians_diff=angle_bc - angle_ba             # Difference between the two angles
     # Step 3: Convert the result from radians to degrees
    angle_in_degrees=np.abs(np.degrees(radians_diff)) # Absolute value in degrees
    # Step 4: Return the angle
    return angle_in_degrees

def get_distance(landmark_list):
    # Check if the list contains at least two landmarks (points)
    if len(landmark_list) < 2:
        return  # Exit the function if not enough points are provided

    # Unpack the first two landmarks into (x1, y1) and (x2, y2)
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    # Calculate the Euclidean distance between the two points
    # Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    L = np.hypot(x2 - x1, y2 - y1)
    # Scale the distance from range [0, 1] to [0, 1000] for better usability
    # This makes the distance more readable or useful for UI/gesture detection
    return np.interp(L, [0, 1], [0, 1000])
