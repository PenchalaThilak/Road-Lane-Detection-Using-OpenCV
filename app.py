import cv2
import numpy as np
import gradio as gr
import tempfile
import os

def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, hough_lines, height):
    if hough_lines is None:
        return image

    filtered_lines = [line for line in hough_lines if max(line[0][1], line[0][3]) > height * 0.6]

    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def process(img):
    height, width = img.shape[:2]
    roi_vertices = np.array([
        (width * 0.1, height),
        (width * 0.4, height * 0.6),
        (width * 0.6, height * 0.6),
        (width * 0.9, height)
    ], np.int32)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)
    roi_img = roi(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=50)
    
    return draw_lines(img, lines, height) if lines is not None else img

def lane_detection(video_path):
    if not os.path.exists(video_path):
        return "Error: Video file not found."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Avoid division by zero
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v for broader compatibility
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        return "Error: Could not create output video file."
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process(frame)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else "Error: Output video file was not created."

demo = gr.Interface(
    fn=lane_detection,
    inputs=gr.Video(label="Original Video"),
    outputs=gr.Video(label="Processed Video"),
    allow_flagging="never"
)

demo.launch()
