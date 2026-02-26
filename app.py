from flask import Flask, render_template, Response, jsonify
import cv2
from stream_processor import VideoCamera

app = Flask(__name__)

# Strict Singleton to ensure only one camera exists across threads
camera_instance = None

def get_camera():
    global camera_instance
    if camera_instance is None:
        print("DEBUG: Creating fresh VideoCamera instance...")
        camera_instance = VideoCamera()
    return camera_instance

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    response = Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/stats')
def emotion_stats():
    """Endpoint to get live emotion statistics for Chart.js"""
    cam = get_camera()
    stats = cam.get_current_stats()
    # Debug print to see what's being sent to the browser
    print(f"DEBUG: Sending stats: {stats}") 
    return jsonify(stats)

@app.route('/api/toggle_emoji', methods=['POST'])
def toggle_emoji():
    """Toggle AR Emoji overlay"""
    cam = get_camera()
    status = cam.toggle_emoji()
    return jsonify({"emoji_enabled": status})

if __name__ == '__main__':
    # use_reloader=False is CRITICAL for OpenCV on Windows to prevent double-initialization
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
