import cv2
import numpy as np
import os

emotions = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 128, 0),    # Dark Green
    "scared": (128, 0, 128),   # Purple
    "happy": (0, 255, 255),    # Yellow
    "sad": (255, 0, 0),        # Blue
    "surprised": (0, 165, 255),# Orange
    "neutral": (128, 128, 128) # Gray
}

def create_emoji(emotion_name, color):
    # Create 100x100 transparent image (4 channels)
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    
    # Draw a colored circle with full alpha
    cv2.circle(img, (50, 50), 45, color + (255,), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, emotion_name[:3].upper(), (20, 55), font, 0.6, (0, 0, 0, 255), 2, cv2.LINE_AA)
    
    # Save to emojis folder
    cv2.imwrite(f"emojis/{emotion_name}.png", img)
    print(f"Created emojis/{emotion_name}.png")

if __name__ == "__main__":
    if not os.path.exists("emojis"):
        os.makedirs("emojis")
        
    for name, color in emotions.items():
        create_emoji(name, color)
