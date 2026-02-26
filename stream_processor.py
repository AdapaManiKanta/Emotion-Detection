import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class VideoCamera(object):
    def __init__(self):
        print("WEB CAMERA: Initializing hardware...")
        # Using OpenCV to capture from device 1 (Device 0 is failing with OS error)
        self.video = cv2.VideoCapture(1)
        import time
        self.start_time = time.time()
        
        # Load the upgraded face detection model (Caffe ResNet SSD)
        self.face_net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Load original emotion classification model
        self.emotion_classifier = load_model("models/_mini_XCEPTION.102-0.66.hdf5", compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        
        # State variables
        self.show_emoji = False
        self.current_stats = {e: 0.0 for e in self.EMOTIONS}
        self.stats_count = 0  # Heartbeat counter to verify data is refreshing
        
        # Load emoticons with Alpha Channel (-1)
        self.emojis = {}
        for emotion in self.EMOTIONS:
            # We assume emojis are stored in a folder called 'emojis'
            # Fallback if no emojis are found: we just won't draw them
            try:
                img = cv2.imread(f"emojis/{emotion}.png", -1)
                if img is not None:
                    self.emojis[emotion] = img
                else:
                    self.emojis[emotion] = None
            except:
                self.emojis[emotion] = None
                
    def __del__(self):
        self.video.release()
        
    def toggle_emoji(self):
        self.show_emoji = not self.show_emoji
        return self.show_emoji
        
    def get_current_stats(self):
        import time
        # Cast to float to avoid JSON serialization issues with NumPy float32
        data = {k: float(v) for k, v in self.current_stats.items()}
        data['count'] = int(self.stats_count)
        data['timestamp'] = time.time()
        return data

    def overlay_emoji(self, background, overlay, x, y, w, h):
        """Helper to overlay a transparent PNG map onto a background image"""
        if overlay is None: return background
        
        overlay = cv2.resize(overlay, (w, h))
        v_h, v_w, _ = background.shape
        
        # Clip bounding box to screen edges
        x1, x2 = max(0, x), min(v_w, x + w)
        y1, y2 = max(0, y), min(v_h, y + h)
        
        # Get overlay slice and alpha channel
        overlay_slice = overlay[y1-y:y2-y, x1-x:x2-x]
        if overlay_slice.shape[2] == 4:
            alpha = overlay_slice[:, :, 3] / 255.0
            for c in range(3):
                background[y1:y2, x1:x2, c] = (alpha * overlay_slice[:, :, c] + 
                                               (1.0 - alpha) * background[y1:y2, x1:x2, c])
        return background

    def get_frame(self):
        success, image = self.video.read()
        if not success or image is None or image.size == 0:
            # Create a blank image with an error message so the UI doesn't freeze or show a black box
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "Webcam not found or blocked.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Please ensure no other app gives it.", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
            
        (h, w) = image.shape[:2]
        
        # Prepare frame for Face Detection (ResNet SSD)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, 
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Reset current frame stats to calculate averages across multiple faces
        frame_emotions = {e: [] for e in self.EMOTIONS}

        # Loop over the detections (Multi-Face tracking)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Clip to frame edges
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                fW, fH = endX - startX, endY - startY
                
                # Ensure valid box
                if fW < 30 or fH < 30: continue
                
                # Extract ROI for emotion classification
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                roi = gray[startY:endY, startX:endX]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                # Predict Emotion
                preds = self.emotion_classifier.predict(roi, verbose=0)[0]
                label_idx = preds.argmax()
                label = self.EMOTIONS[label_idx]
                
                # Store for aggregate stats
                for idx, em in enumerate(self.EMOTIONS):
                    frame_emotions[em].append(preds[idx] * 100)

                if self.show_emoji and self.emojis[label] is not None:
                    # Draw AR Emoji with Alpha Blending
                    image = self.overlay_emoji(image, self.emojis[label], startX, startY, fW, fH)
                else:
                    # Draw standard bounding box and text
                    text = f"{label}: {preds[label_idx]*100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Update global stats with averages from this frame
        found_faces = any(len(lst) > 0 for lst in frame_emotions.values())
        
        if found_faces:
            self.stats_count += 1
            for em in self.EMOTIONS:
                if frame_emotions[em]:
                    # Explicitly cast to float and round to 2 decimals
                    val = float(np.mean(frame_emotions[em]))
                    self.current_stats[em] = round(val, 2)
                else:
                    self.current_stats[em] = 0.0
        else:
            # Decay stats to zero if no faces found (slower decay for smoother UI)
            for em in self.EMOTIONS:
                current_val = float(self.current_stats.get(em, 0.0))
                self.current_stats[em] = round(max(0.0, current_val - 1.0), 2)

        # Encode frame for HTTP streaming
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
