# PiProjects-HomeAutomation
# Smart Home Control Hub

A multi-application system running on Raspberry Pi 5 that provides various interfaces for controlling smart home routines. The system consists of two main components:

1. A gesture recognition interface that uses computer vision to control smart home routines through hand gestures
2. A web-based control interface with system monitoring, button controls, and Meraki webhook integration

Both applications can trigger Alexa routines via VoiceMonkey and interact with Home Assistant webhooks.

## Features

### Gesture Recognition Application (Port 5555)
- Real-time hand gesture recognition using MediaPipe
- Support for multiple cameras with different gesture mappings
- Live video feed visualization with gesture overlay
- Systemd logging integration
- Frame preprocessing for improved recognition in various lighting conditions

### Web Control Application (Port 5050)
- System monitoring dashboard showing:
  - CPU usage
  - RAM usage
  - Storage information
  - Compute temperature
- Password-protected button control interface
- Meraki button webhook integration
- Multi-device stats collection and display
- VoiceMonkey integration for Alexa routine control

### Shared Features
- Integration with Alexa routines via VoiceMonkey
- Home Assistant webhook integration
- Running on Raspberry Pi 5 hardware
- Secure environment variable configuration

## Prerequisites

### Hardware
- Raspberry Pi 5 (8GB model)
- Compatible USB webcams
- Python 3.x

### Software Dependencies
All required Python packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your configuration:
```env
# Home Assistant Webhooks
HOME_ASSISTANT_WEBHOOK_ARRIVE_ID=your_arrive_webhook_id
HOME_ASSISTANT_WEBHOOK_LEAVE_ID=your_leave_webhook_id
HOME_ASSISTANT_URL=http://your-ha-instance.local
HOME_ASSISTANT_WEBHOOK_ARRIVE_ID=your_arrive_webhook_id
HOME_ASSISTANT_WEBHOOK_LEAVE_ID=your_leave_webhook_id
FLASK_HOST=your_host_ip
FLASK_PORT=5555

# MediaPipe COnfiguration
GESTURE_MODEL_PATH=/path/to/your/gesture_recognizer.task

# VoiceMonkey Configuration
ROUTINE_API_TOKEN=your_voicemonkey_token

# Web Interface Security
SECRET_KEY=your_flask_secret_key
BUTTON_PAGE_PASSWORDS=password1,password2,password3
HOST=your_host_ip
PORT=5050

# Meraki Configuration
MERAKI_API_TOKEN=your_meraki_api_token
```

4. Download the MediaPipe gesture recognizer model:
```bash
# Ensure the model is placed at:
/home/[username]/SW_PROJECTS/MT30ALEXAFLASK/gesture_recognizer.task
```

## Configuration

### Gesture Mappings

The application supports different gesture mappings for each camera. Current supported gestures include:

- Peace Sign
- Open Hand
- Rock On (ILoveYou gesture)
- Fist
- Pointing Up
- Thumbs Up/Down

Gestures can be mapped to different actions based on which hand (Left/Right) performs them.

### Camera Configuration

Two cameras are supported by default:
- Camera 1 (`/dev/video0`): Front-facing with frame preprocessing and flipping
- Camera 2 (`/dev/video2`): Secondary camera without preprocessing

### Current Gesture Mappings

#### Camera 1:
- Right Peace Sign → Turn off lights + Trigger "Leave" webhook
- Left Open Hand → White lights + Trigger "Arrive" webhook

#### Camera 2:
- Right Rock On → Movie mode
- Left Rock On → White lights
- Left Peace Sign → Lights off

## Usage

1. Start both applications (in separate terminal sessions):
```bash
# Terminal 1 - Gesture Recognition
python gesture_control.py

# Terminal 2 - Web Control Interface
python app.py
```

2. Access the web interfaces:
```
# Gesture Recognition Interface
http://192.168.100.55:5555

# System Monitoring and Control Interface
http://192.168.100.55:5050
```

### Web Interface Features

#### System Monitoring
- Access the main dashboard at `/` to view system statistics
- Real-time updates of CPU, RAM, storage, and temperature
- Support for collecting stats from multiple devices

#### Button Control Interface
1. Navigate to `/button-control`
2. Log in using one of the configured passwords
3. Access buttons for:
   - All Off
   - White Lights
   - Lights Off Only

#### Meraki Integration
- Webhook endpoint at `/meraki_webhook`
- Supports both short and long button presses:
  - Short press: Triggers "All Off" routine
  - Long press: Triggers "White Lights" routine

The interface shows live feeds from both cameras and displays recognized gestures with their confidence scores.

## Performance Optimization

The application includes several optimizations:
- Frame skipping (processes every 3rd frame)
- Reduced frame resolution (640x480)
- JPEG compression for web streaming
- Minimal frame buffer size
- Optional frame preprocessing for difficult lighting conditions

## Logging

The application logs to the systemd journal. View logs using:
```bash
journalctl -u gesture-control
```

## Error Handling

- Automatic recovery from camera read errors
- Gesture confidence thresholds to prevent false positives
- Connection timeout handling for webhook calls

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

No license!

## Acknowledgments

- Google MediaPipe for the gesture recognition model
- VoiceMonkey for Alexa routine integration
- Home Assistant for webhook functionality
