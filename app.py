from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open camera.")

def generate_frames():
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
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

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
    return render_template('index.html')  # Create a simple HTML page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8080)


