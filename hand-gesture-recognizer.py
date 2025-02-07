"""
Gesture Recognition System for Smart Home Control
Uses MediaPipe for hand gesture recognition to control smart home devices via 
Alexa routines and Home Assistant webhooks. Supports multiple cameras and 
provides a web interface for monitoring.

Requirements:
- Python 3.x
- OpenCV
- MediaPipe
- Flask
- Required environment variables (see README)

Author: Your Name
License: Your License
"""

import cv2
import numpy as np
import math
import mediapipe as mp
import threading
import logging
import time
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from systemd import journal
from mediapipe.tasks import python as tasks
from mediapipe.tasks import python as vision
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
from flask import Flask, Response, render_template
from app import triggerAlexaOffRoutine, triggerAlexaWhiteRoutine, triggerAlexaMovieRoutine, triggerAlexaLightsOnlyOffRoutine

# Load environment variables
load_dotenv()

# Configure logging to use systemd journal
logger = logging.getLogger('gesture-control')
logger.setLevel(logging.INFO)
journald_handler = journal.JournalHandler()
journald_handler.setFormatter(logging.Formatter(
    '%(levelname)s - %(message)s'
))
logger.addHandler(journald_handler)

# Custom gesture names mapping
custom_gesture_names = {
    "Unknown": "No Gesture",
    "Closed_Fist": "Fist",
    "Open_Palm": "Open Hand",
    "Pointing_Up": "Point Up",
    "Thumb_Down": "Dislike",
    "Thumb_Up": "Like",
    "Victory": "Peace",
    "ILoveYou": "Rock On"
}

def trigger_webhook(webhook_url):
    """Trigger a webhook with error handling."""
    try:
        response = requests.post(webhook_url, timeout=5)
        response.raise_for_status()
        logger.info(f"Webhook triggered successfully: {webhook_url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to trigger webhook {webhook_url}: {str(e)}")

def execute_actions(actions):
    """Execute routine and/or webhook actions."""
    if "routine" in actions:
        actions["routine"]()
    if "webhook" in actions:
        trigger_webhook(actions["webhook"])

def preprocess_frame(frame, alpha=1.2, beta=-30, gamma=.8, apply_CLAHE=True, apply_GAUSSIAN=True, apply_gray_threshold=True):
    """Preprocess video frame for better gesture recognition."""
    # Initial brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply gamma correction
    adjusted = np.array(255 * (adjusted / 255) ** gamma, dtype='uint8')
    
    if apply_CLAHE:
        # Adaptive histogram equalization for better dynamic range
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        adjusted = cv2.merge((cl,a,b))
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
    
    if apply_GAUSSIAN:
        # Apply mild Gaussian blur to reduce noise
        adjusted = cv2.GaussianBlur(adjusted, (3,3), 0)
    
    if apply_gray_threshold:
        # Threshold very bright areas
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)
        adjusted = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return adjusted

class GestureProcessor:
    """Handles video capture and gesture recognition for a single camera."""
    
    def __init__(self, camera_id, gesture_config, enable_preprocessing=True, flip_frame=False):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.is_running = False
        self.thread = None
        self.enable_preprocessing = enable_preprocessing
        self.flip_frame = flip_frame
        self.last_action = "No action triggered"
        self.last_action_time = None
        
        # Frame rate control
        self.process_every_n_frames = 3  # Only process every 3rd frame
        self.frame_counter = 0
        
        # Camera properties
        self.frame_width = 640
        self.frame_height = 480
        
        logger.info(f"Initializing GestureProcessor for camera {camera_id}")
        try:
            # Initialize MediaPipe and models
            self.model_path = os.environ.get('GESTURE_MODEL_PATH', './gesture_recognizer.task')
            self.base_options = vision.BaseOptions(
                model_asset_path=self.model_path,
                delegate=tasks.BaseOptions.Delegate.CPU
            )
            self.options = GestureRecognizerOptions(
                base_options=self.base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=2
            )
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=0
            )
            logger.info(f"MediaPipe models initialized successfully for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe models for camera {camera_id}: {str(e)}")
            raise

        # Initialize gesture configuration
        self.gesture_actions = gesture_config
        self.gesture_timers = {gesture: {"start_time": None, "active": False} 
                             for gesture in self.gesture_actions}

    def start(self):
        """Start the gesture processor."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.thread = threading.Thread(target=self.process_frames)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"GestureProcessor started successfully for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to start GestureProcessor for camera {self.camera_id}: {str(e)}")
            raise

    def stop(self):
        """Stop the gesture processor."""
        logger.info(f"Stopping GestureProcessor for camera {self.camera_id}")
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info(f"GestureProcessor stopped successfully for camera {self.camera_id}")

    def get_frame(self):
        """Return the current frame."""
        return self.frame

    def process_frames(self):
        """Main processing loop for video frames."""
        error_count = 0
        
        with GestureRecognizer.create_from_options(self.options) as recognizer:
            logger.info(f"Starting frame processing loop for camera {self.camera_id}")
            while self.is_running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        error_count += 1
                        if error_count >= 5:
                            logger.error(f"Multiple consecutive failures to read frame from camera {self.camera_id}")
                            error_count = 0
                        continue

                    # Skip frames based on counter
                    self.frame_counter += 1
                    if self.frame_counter % self.process_every_n_frames != 0:
                        self.frame = frame
                        continue

                    # Resize frame if needed
                    if frame.shape[1] > self.frame_width or frame.shape[0] > self.frame_height:
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                    # Preprocess the frame if enabled
                    if self.enable_preprocessing:
                        frame = preprocess_frame(
                            frame, 
                            alpha=1.1, 
                            beta=-35, 
                            gamma=0.6, 
                            apply_CLAHE=True, 
                            apply_GAUSSIAN=True, 
                            apply_gray_threshold=False
                        )
                    
                    # Flip frame if enabled
                    process_frame = cv2.flip(frame, -1) if self.flip_frame else frame
                    frame_timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                    
                    # Convert to MediaPipe format
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    )
                    
                    # Perform gesture recognition
                    gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
                    
                    # Process recognized gestures with handedness
                    if gesture_recognition_result.gestures and gesture_recognition_result.handedness:
                        for gesture, handedness in zip(gesture_recognition_result.gestures,
                                                     gesture_recognition_result.handedness):
                            if gesture[0].score > 0.5:
                                category_name = gesture[0].category_name or "Unknown"
                                custom_label = custom_gesture_names.get(category_name, category_name)
                                hand_side = "Right" if handedness[0].category_name == "Right" else "Left"
                                
                                gesture_tuple = (custom_label, hand_side)
                                
                                if gesture_tuple in self.gesture_actions:
                                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                                    
                                    if not self.gesture_timers[gesture_tuple]["active"]:
                                        logger.info(f"Detected {hand_side} hand gesture: {custom_label} on camera {self.camera_id}")
                                        self.gesture_timers[gesture_tuple]["start_time"] = current_time
                                        self.gesture_timers[gesture_tuple]["active"] = True
                                    else:
                                        elapsed_time = current_time - self.gesture_timers[gesture_tuple]["start_time"]
                                        if elapsed_time >= self.gesture_actions[gesture_tuple]["duration"]:
                                            try:
                                                self.gesture_actions[gesture_tuple]["action"]()
                                                self.last_action = f"{hand_side} {custom_label}"
                                                self.last_action_time = datetime.now()
                                                logger.info(f"Successfully triggered action for {hand_side} {custom_label} on camera {self.camera_id}")
                                            except Exception as e:
                                                logger.error(f"Failed to trigger action for {hand_side} {custom_label} on camera {self.camera_id}: {str(e)}")
                                            
                                            self.gesture_timers[gesture_tuple]["active"] = False
                                            self.gesture_timers[gesture_tuple]["start_time"] = None

                            # Display gesture info with handedness
                            if gesture[0].score > 0.7:
                                cv2.putText(
                                    frame,
                                    f"{hand_side} {custom_label}: {gesture[0].score:.2f}",
                                    (10, 30 if hand_side == "Right" else 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2
                                )

                    # Reset inactive gestures
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    for gesture_tuple in list(self.gesture_timers.keys()):
                        if self.gesture_timers[gesture_tuple]["active"]:
                            if current_time - self.gesture_timers[gesture_tuple]["start_time"] > \
                               self.gesture_actions[gesture_tuple]["duration"]:
                                self.gesture_timers[gesture_tuple]["active"] = False
                                self.gesture_timers[gesture_tuple]["start_time"] = None

                    # Process hand landmarks
                    if self.frame_counter % self.process_every_n_frames == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(rgb_frame)

                        if results.multi_hand_landmarks:
                            for landmarks in results.multi_hand_landmarks:
                                # Draw landmarks
                                for i in range(21):
                                    x = int(landmarks.landmark[i].x * frame.shape[1])
                                    y = int(landmarks.landmark[i].y * frame.shape[0])
                                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                                # Draw connections
                                connections = self.mp_hands.HAND_CONNECTIONS
                                for connection in connections:
                                    start_idx, end_idx = connection
                                    start = landmarks.landmark[start_idx]
                                    end = landmarks.landmark[end_idx]
                                    start_point = (int(start.x * frame.shape[1]), 
                                                 int(start.y * frame.shape[0]))
                                    end_point = (int(end.x * frame.shape[1]), 
                                               int(end.y * frame.shape[0]))
                                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                    self.frame = frame

                except Exception as e:
                    logger.error(f"Error in frame processing loop for camera {self.camera_id}: {str(e)}")
                    error_count += 1
                    if error_count >= 5:
                        logger.critical(f"Multiple consecutive errors in frame processing for camera {self.camera_id}, but continuing...")
                        error_count = 0
                    time.sleep(0.1)

# Environment variables for webhooks
WEBHOOK_ID_ARRIVE = os.environ['HOME_ASSISTANT_WEBHOOK_ARRIVE_ID']
WEBHOOK_ID_LEAVE = os.environ['HOME_ASSISTANT_WEBHOOK_LEAVE_ID']
HOME_ASSISTANT_URL = os.environ.get('HOME_ASSISTANT_URL', 'http://your-ha-instance.local')

# Define gesture configurations for each camera
camera1_gestures = {
    ("Peace", "Right"): {
        "action": lambda: execute_actions({
            "routine": triggerAlexaOffRoutine,
            "webhook": f"{HOME_ASSISTANT_URL}/api/webhook/{WEBHOOK_ID_LEAVE}"
        }),
        "duration": 0.2
    },
    ("Open Hand", "Left"): {
        "action": lambda: execute_actions({
            "routine": triggerAlexaWhiteRoutine,
            "webhook": f"{HOME_ASSISTANT_URL}/api/webhook/{WEBHOOK_ID_ARRIVE}"
        }),
        "duration": 0.2
    }
}

camera2_gestures = {
    ("Rock On", "Right"): {
        "action": lambda: execute_actions({
            "routine": triggerAlexaMovieRoutine
        }),
        "duration": 0.2
    },
    ("Rock On", "Left"): {
        "action": lambda: execute_actions({
            "routine": triggerAlexaWhiteRoutine
        }),
        "duration": 0.2
    },
    ("Peace", "Left"): {
        "action": lambda: execute_actions({
            "routine": triggerAlexaLightsOnlyOffRoutine
        }),
        "duration": 0.2
    }
}

# Initialize gesture processors for both cameras
gesture_processor1 = GestureProcessor("/dev/video0", camera1_gestures, enable_preprocessing=True, flip_frame=True)
gesture_processor2 = GestureProcessor("/dev/video2", camera2_gestures, enable_preprocessing=False, flip_frame=False)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('handgesture.html')

@app.route('/last_actions')
def get_last_actions():
    return {
        'camera1': {
            'last_action': gesture_processor1.last_action,
            'timestamp': gesture_processor1.last_action_time.isoformat() if gesture_processor1.last_action_time else None
        },
        'camera2': {
            'last_action': gesture_processor2.last_action,
            'timestamp': gesture_processor2.last_action_time.isoformat() if gesture_processor2.last_action_time else None
        }
    }

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    def generate_frames(processor):
        while True:
            frame = processor.get_frame()
            if frame is not None:
                # Reduce image quality for transmission
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)  # Add small delay when no frame is available

    processor = gesture_processor1 if camera_id == '1' else gesture_processor2
    return Response(generate_frames(processor),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logger.info("Starting Gesture Recognition Application")
    gesture_processor1.start()
    gesture_processor2.start()
    try:
        # Use environment variables for host and port
        host = os.environ.get('FLASK_HOST', '0.0.0.0')
        port = int(os.environ.get('FLASK_PORT', 5555))
        app.run(host=host, port=port)
    except Exception as e:
        logger.critical(f"Flask application failed: {str(e)}")
    finally:
        logger.info("Shutting down Gesture Recognition Application")
        gesture_processor1.stop()
        gesture_processor2.stop()
