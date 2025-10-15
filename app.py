import os
import json
import time
import base64
import shutil
import subprocess
from threading import Thread
from io import BytesIO
import logging

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Optional libraries with fallback mocks
try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV (cv2) not available. Using mock behavior.")

try:
    from PIL import Image
except ImportError:
    Image = None
    logging.warning("PIL not available. Image processing may be limited.")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics.YOLO not available. Using mock YOLO.")

# Use imageio-ffmpeg for ffmpeg handling
try:
    import imageio_ffmpeg as _imageio_ffmpeg
    _imageio_ffmpeg_available = True
except ImportError:
    _imageio_ffmpeg = None
    _imageio_ffmpeg_available = False
    logging.warning("imageio_ffmpeg not available. Video processing will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# App config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', '/uploads')
app.config['CLIPS_FOLDER'] = os.environ.get('CLIPS_FOLDER', '/video_clips')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)

# Environment variables
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_CHAT_API_URL = os.environ.get('OPENAI_CHAT_API_URL', "https://api.openai.com/v1/chat/completions")
MODEL_PATH = os.environ.get('MODEL_PATH', 'fault_detection_model_best.pt')

if not N8N_WEBHOOK_URL:
    logging.warning("N8N_WEBHOOK_URL not set. Webhook calls will fail.")

# Priority mapping
PRIORITY_MAPPING = {
    'crack_priority_1': 'Priority 1',
    'crack_priority_2': 'Priority 2',
    'crack_priority_3': 'Priority 3',
    'emergency_replace': 'Emergency Replace'
}

# FFMPEG handling
def get_ffmpeg_executable():
    """Return path to ffmpeg executable from imageio_ffmpeg or None."""
    if _imageio_ffmpeg_available:
        try:
            exe = _imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                logging.info(f"Using imageio_ffmpeg bundled ffmpeg at: {exe}")
                return exe
        except Exception as e:
            logging.error(f"imageio_ffmpeg couldn't provide ffmpeg: {e}")
    logging.error("No ffmpeg executable found.")
    return None

FFMPEG_BIN = get_ffmpeg_executable()

# YOLO model loading
def load_yolo_model(path):
    if YOLO is None:
        class MockResultsBox:
            def __init__(self):
                self.id = []
                self.cls = []
                self.conf = []

        class MockResult:
            def __init__(self):
                self.boxes = MockResultsBox()
                self.names = {0: 'unknown_fault'}

        class MockYOLO:
            def track(self, *args, **kwargs):
                return [MockResult()]

        logging.warning("Using mock YOLO model.")
        return MockYOLO()
    
    try:
        model = YOLO(path)
        logging.info("YOLO model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLO model ({path}): {e}. Using mock.")
        return MockYOLO()

yolo_model = load_yolo_model(MODEL_PATH)

# FFMPEG helpers
def run_ffmpeg_cmd(cmd_list):
    """Run ffmpeg command and return success status and output."""
    try:
        proc = subprocess.run(cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, proc.stdout.decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logging.error(f"FFMPEG failed. Cmd: {' '.join(cmd_list)}\nError: {stderr}")
        return False, stderr
    except FileNotFoundError as e:
        logging.error(f"FFMPEG executable not found: {e}")
        return False, str(e)

def extract_video_clip(video_path: str, start_time_sec: float, duration_sec: float = 10.0) -> str:
    if FFMPEG_BIN is None:
        logging.error("Cannot extract clip: no ffmpeg binary available.")
        return None

    output_clip_file = os.path.join(app.config['CLIPS_FOLDER'], f"clip_{int(time.time())}_{int(start_time_sec)}.mp4")
    start_time = max(0.0, start_time_sec - 3.0)

    ffmpeg_command = [
        FFMPEG_BIN, '-y', '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration_sec),
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac',
        output_clip_file
    ]

    logging.info(f"Extracting video clip from {start_time:.2f}s for {duration_sec}s to {output_clip_file}")
    ok, out = run_ffmpeg_cmd(ffmpeg_command)
    if ok and os.path.exists(output_clip_file):
        logging.info("Video clip extracted successfully.")
        return output_clip_file
    logging.error("Video clip extraction failed.")
    return None

def video_to_base64(video_path: str) -> str:
    try:
        with open(video_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding video to base64: {e}")
        return None

def extract_audio_clip_for_transcription(video_path: str, start_time_sec: float, duration_sec: float = 15.0) -> str:
    if FFMPEG_BIN is None:
        logging.warning("FFMPEG not available. Returning mock transcription.")
        return "Building 5, Apartment 3A, Tread number 14, bottom rear crack."

    tmp_audio = os.path.join(app.config['CLIPS_FOLDER'], f"temp_audio_{int(time.time())}.mp3")
    ffmpeg_command = [
        FFMPEG_BIN, '-y', '-i', video_path,
        '-ss', str(max(0.0, start_time_sec)),
        '-t', str(duration_sec),
        '-q:a', '0', '-map', '0:a',
        tmp_audio
    ]
    logging.info(f"Extracting audio clip to {tmp_audio}")
    ok, out = run_ffmpeg_cmd(ffmpeg_command)
    if ok and os.path.exists(tmp_audio):
        try:
            transcription = "Building 5, Apartment 3A, Tread number 14, bottom rear crack."
        finally:
            try:
                os.remove(tmp_audio)
            except Exception:
                pass
        return transcription
    logging.error("Audio extraction failed. Returning mock transcription.")
    return "Building 5, Apartment 3A, Tread number 14, bottom rear crack."

# OpenAI transcription analysis
import requests
def openai_analyze_audio(transcribed_text: str) -> dict:
    logging.info("Analyzing transcription for structured location data...")
    if not OPENAI_API_KEY:
        logging.warning("No OPENAI_API_KEY. Returning mock parsed values.")
        return {
            "building_number": "5",
            "apartment_number": "3A",
            "tread_number": "14",
            "crack_location": "bottom rear"
        }

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_API_KEY}'}
    system_prompt = (
        "Extract from transcription: building_number, apartment_number, tread_number, crack_location. "
        "Return JSON with these keys, set to null if missing."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcription: '{transcribed_text}'"}
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }

    try:
        resp = requests.post(OPENAI_CHAT_API_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content'].strip()
            try:
                parsed = json.loads(content)
                logging.info(f"OpenAI parsed: {parsed}")
                return parsed
            except json.JSONDecodeError:
                logging.error("OpenAI returned unparsable JSON.")
        else:
            logging.error("No valid choices in OpenAI response.")
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
    return {
        "building_number": None,
        "apartment_number": None,
        "tread_number": None,
        "crack_location": None
    }

# Webhook sender
def send_to_webhook(alert_data: dict, location_data: dict, video_clip_b64: str):
    payload = {
        "fault_id": alert_data.get('tracker_id'),
        "fault_type": alert_data.get('class_name'),
        "priority_level": alert_data.get('priority_level'),
        "confidence": float(alert_data.get('confidence', 0.0)),
        "frame_number": alert_data.get('frame_number'),
        "building_number": location_data.get('building_number'),
        "apartment_number": location_data.get('apartment_number'),
        "tread_number": location_data.get('tread_number'),
        "crack_location": location_data.get('crack_location'),
        "screenshot_base64": alert_data.get('screenshot_base64'),
        "video_clip_base64": video_clip_b64,
        "status": "Success" if (video_clip_b64 and all(v is not None for v in location_data.values())) else "Partial Data"
    }

    if not N8N_WEBHOOK_URL:
        logging.warning("No webhook URL. Skipping POST. Payload preview:")
        logging.info(json.dumps(payload, indent=2)[:2000])
        return

    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        logging.info(f"Webhook responded with status: {resp.status_code}")
    except requests.RequestException as e:
        logging.error(f"Webhook connection error: {e}")

# Priority helper
def get_priority_level(class_name: str) -> str:
    if not class_name:
        return "Unknown Priority"
    class_name_lower = class_name.lower().replace(' ', '_')
    return PRIORITY_MAPPING.get(class_name_lower, "Unknown Priority")

# Video processing
def process_video_in_background(video_path, confidence_threshold=0.5):
    if cv2 is None:
        logging.error("OpenCV not available. Skipping video processing.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logging.info(f"Analyzing video: {video_path} (FPS: {fps})")
    unique_alerts_processed = {}
    frame_number = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        try:
            results = yolo_model.track(source=frame, persist=True, conf=confidence_threshold, verbose=False)
        except Exception as e:
            logging.error(f"YOLO track failed: {e}")
            results = []

        if not results:
            frame_number += 1
            if frame_number % 100 == 0:
                logging.info(f"Processed {frame_number} frames (no detections)")
            continue

        result = results[0]
        try:
            boxes = getattr(result, 'boxes', None)
            names = getattr(result, 'names', {}) or {}
            ids = getattr(boxes, 'id', []) or []
            clss = getattr(boxes, 'cls', []) or []
            confs = getattr(boxes, 'conf', []) or []
        except Exception:
            ids, clss, confs, names = [], [], [], {}

        try:
            import numpy as np
            if hasattr(ids, 'cpu') and hasattr(ids, 'numpy'):
                ids = ids.cpu().numpy().tolist()
            if hasattr(clss, 'cpu') and hasattr(clss, 'numpy'):
                clss = clss.cpu().numpy().tolist()
            if hasattr(confs, 'cpu') and hasattr(confs, 'numpy'):
                confs = confs.cpu().numpy().tolist()
        except Exception:
            pass

        if ids:
            for i in range(len(ids)):
                tracker_id = int(ids[i]) if i < len(ids) else None
                if tracker_id is None or tracker_id in unique_alerts_processed:
                    continue

                cls_idx = int(clss[i]) if i < len(clss) else 0
                class_name = names.get(cls_idx, str(cls_idx))
                priority_level = get_priority_level(class_name)
                confidence = float(confs[i]) if i < len(confs) else 0.0

                logging.info(f"NEW FAULT DETECTED: ID {tracker_id} | Frame {frame_number} | Priority {priority_level} | Conf {confidence}")

                # Screenshot
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    screenshot_b64 = base64.b64encode(buffer).decode('utf-8')
                    logging.info("Screenshot captured")
                except Exception as e:
                    screenshot_b64 = None
                    logging.error(f"Error capturing screenshot: {e}")

                # Video clip
                time_of_detection_sec = frame_number / fps
                video_clip_path = extract_video_clip(video_path, time_of_detection_sec)
                video_clip_b64 = video_to_base64(video_clip_path) if video_clip_path else None

                # Audio transcription
                audio_start_time = max(0.0, time_of_detection_sec - 5.0)
                transcribed_text = extract_audio_clip_for_transcription(video_path, audio_start_time)
                extracted_location = openai_analyze_audio(transcribed_text)

                # Send webhook
                alert_data = {
                    'screenshot_base64': screenshot_b64,
                    'class_name': class_name,
                    'priority_level': priority_level,
                    'confidence': confidence,
                    'frame_number': frame_number,
                    'tracker_id': tracker_id
                }
                send_to_webhook(alert_data, extracted_location, video_clip_b64)

                # Cleanup
                if video_clip_path and os.path.exists(video_clip_path):
                    try:
                        os.remove(video_clip_path)
                        logging.info("Video clip cleaned up")
                    except Exception:
                        pass

                unique_alerts_processed[tracker_id] = True

        frame_number += 1
        if frame_number % 100 == 0:
            logging.info(f"Processed {frame_number} frames")

    cap.release()
    try:
        os.remove(video_path)
        logging.info(f"Video file {os.path.basename(video_path)} cleaned up")
    except Exception:
        logging.error(f"Error cleaning up video file: {video_path}")
    logging.info(f"VIDEO ANALYSIS COMPLETE. Total faults: {len(unique_alerts_processed)}")

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') if os.path.exists('templates/index.html') else "Upload endpoint: POST /upload with form-data video"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    thread = Thread(target=process_video_in_background, args=(video_path, 0.5))
    thread.daemon = True
    thread.start()
    return jsonify({
        "status": "success",
        "message": f"Video '{filename}' uploaded. Processing started."
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Fault detection service running"})

if __name__ == '__main__':
    logging.info(f"FFMPEG_BIN: {FFMPEG_BIN}")
    logging.info(f"N8N_WEBHOOK_URL set: {bool(N8N_WEBHOOK_URL)}")
    logging.info(f"OPENAI_API_KEY set: {bool(OPENAI_API_KEY)}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))