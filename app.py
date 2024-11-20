from flask import Flask, Response, render_template, request, jsonify
import cv2
import os
from ultralytics import YOLO

# Initialize the Flask app
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Global variables for video source
cap = None
video_source = './static/sample_video.mp4'  # Default to pre-recorded video

#helo

def initialize_video_source():
    """Initialize video capture based on the selected source."""
    global cap, video_source
    if cap:
        cap.release()  # Release any previous video source
    if video_source == 'camera':
        cap = cv2.VideoCapture(0)  # Real-time camera
    else:
        cap = cv2.VideoCapture(video_source)  # Uploaded or pre-recorded video

    if not cap.isOpened():
        raise Exception(f"Error: Could not open video source ({video_source}).")


def generate_frames():
    """Yield frames with object detection results."""
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)

        human_count = 0
        total_objects = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                label = model.names[class_id]
                label_text = f"{label} {confidence:.2f}"

                if label.lower() == 'person':
                    human_count += 1
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)
                total_objects += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Overlay counts on the frame
        cv2.putText(frame, f'Total Humans: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Total Objects: {total_objects}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')  # Render the updated HTML page


@app.route('/video_feed')
def video_feed():
    initialize_video_source()  # Initialize the selected video source
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_source', methods=['POST'])
def set_source():
    """Set the video source and restart the stream."""
    global video_source
    data = request.json
    if data.get('source') == 'camera':
        video_source = 'camera'
    elif data.get('source') == 'uploaded' and 'filename' in data:
        video_source = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    else:
        return jsonify({'error': 'Invalid source or missing parameters'}), 400

    # Return success response
    return jsonify({'message': 'Source updated successfully'})


@app.route('/upload', methods=['POST'])
def upload():
    """Handle video file upload and set it as the video source."""
    global video_source
    if 'videoFile' not in request.files:
        return "No file uploaded", 400
    file = request.files['videoFile']
    if file.filename == '':
        return "No file selected", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)  # Save the uploaded file
    video_source = filepath  # Set the uploaded file as the new video source
    return jsonify({'filename': file.filename, 'message': 'File uploaded successfully'})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
