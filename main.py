import cv2
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

object_count = 0
show_graphs = False


def recognize_color(frame, x, y, width, height):
    global object_count

    top_left = (x, y)
    bottom_right = (x + width, y + height)

    diagonal1 = (top_left[0], top_left[1])
    diagonal2 = (bottom_right[0], bottom_right[1])

    cropped_frame = frame[diagonal1[1]:diagonal2[1], diagonal1[0]:diagonal2[0]]

    hsv_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'green': (np.array([35, 100, 100]), np.array([85, 255, 255])),
        'blue': (np.array([100, 100, 100]), np.array([140, 255, 255])),
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 255, 30]))
    }

    recognized_color = None
    avg_color = cv2.mean(cropped_frame)

    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        mask = cv2.inRange(hsv_cropped, lower_bound, upper_bound)
        color_percentage = cv2.mean(mask)[0] / 255.0

        if color_percentage > 0.1:
            if recognized_color != color_name:
                recognized_color = color_name
                object_count += 1
            break

    hsv_values = hsv_cropped[height // 2, width // 2]

    return recognized_color, avg_color, hsv_values, (x + width // 2, y + height // 2)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0
start_time = cv2.getTickCount()

log_file = open('object_info_log.csv', 'a', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Color', 'Width', 'Height', 'X Position', 'Y Position', 'Timestamp'])

object_count_history = []
recognized_colors = []

average_color_intensity = []
h_values = []
s_values = []
v_values = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    x, y, width, height = 100, 100, 200, 150

    recognized_color, avg_color, hsv_values, roi_center = recognize_color(frame, x, y, width, height)

    crosshair_color = (0, 0, 255)
    crosshair_thickness = 2
    crosshair_length = 20
    crosshair_center = (x + width // 2, y + height // 2)
    cv2.line(frame, (crosshair_center[0] - crosshair_length, crosshair_center[1]),
             (crosshair_center[0] + crosshair_length, crosshair_center[1]), crosshair_color, crosshair_thickness)
    cv2.line(frame, (crosshair_center[0], crosshair_center[1] - crosshair_length),
             (crosshair_center[0], crosshair_center[1] + crosshair_length), crosshair_color, crosshair_thickness)

    roi_text = f"ROI Center: ({roi_center[0]}, {roi_center[1]})"
    cv2.putText(frame, roi_text, (x + width + 10, y + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if recognized_color:
        color_text = f"Color: {recognized_color}"
        color_rect_position = (x + width + 10, y + 30)
    else:
        color_text = "Color: Unknown"
        color_rect_position = (x + width + 10, y + 30)

    cv2.putText(frame, color_text, color_rect_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    hsv_text = f"HSV: {hsv_values[0]:.2f}, {hsv_values[1]:.2f}, {hsv_values[2]:.2f}"
    cv2.putText(frame, hsv_text, (x + width + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    object_count_text = f"Objects Detected: {object_count}"
    cv2.putText(frame, object_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    roi = frame[y:y + height, x:x + width]
    cv2.imshow('detection range', roi)

    frame_count += 1
    end_time = cv2.getTickCount()
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
    fps = frame_count / elapsed_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    status_text = f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if recognized_color:
        object_info = [recognized_color, width, height, roi_center[0], roi_center[1], time.time()]
        csv_writer.writerow(object_info)

    object_count_history.append(object_count)
    recognized_colors.append(recognized_color)

    average_intensity = avg_color[0]
    average_color_intensity.append(average_intensity)

    h_values.append(hsv_values[0])
    s_values.append(hsv_values[1])
    v_values.append(hsv_values[2])

    cv2.imshow('main', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 4))
plt.plot(object_count_history)
plt.xlabel('Frame')
plt.ylabel('Object Count')
plt.title('Object Count Over Time')
plt.savefig('object_count_graph.png')

color_counts = {color: recognized_colors.count(color) for color in set(recognized_colors)}
colors = list(color_counts.keys())
counts = list(color_counts.values())

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=colors, autopct='%1.1f%%', startangle=140)
plt.title('Color Distribution')
plt.axis('equal')
plt.savefig('color_distribution_graph.png')

plt.figure(figsize=(10, 4))
plt.plot(average_color_intensity)
plt.xlabel('Frame')
plt.ylabel('Average Color Intensity')
plt.title('Average Color Intensity Over Time')
plt.savefig('average_color_intensity_graph.png')

plt.figure(figsize=(10, 4))
plt.plot(h_values, label='Hue')
plt.plot(s_values, label='Saturation')
plt.plot(v_values, label='Value')
plt.xlabel('Frame')
plt.ylabel('HSV Values')
plt.title('HSV Values Over Time')
plt.legend()
plt.savefig('hsv_values_graph.png')
