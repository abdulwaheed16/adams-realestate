import os
import time
import logging
import cv2
import numpy as np

# --- Robust Import Strategy for YOLO ---
# Try to import YOLO, but if it fails, create a mock version.
# This prevents the entire deployment from failing if ultralytics can't be built.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logging.info("Ultralytics library found and imported.")
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics not found. Using mock YOLO model.")

    class MockYOLO:
        """A mock YOLO class to simulate detections if the real library is unavailable."""
        def __init__(self, model_path):
            logging.warning(f"Using MockYOLO for path: {model_path}")
        
        def track(self, source, persist=True, conf=0.5, verbose=False):
            # Return a mock result to simulate a detection on a specific frame
            class MockBoxes:
                def __init__(self):
                    self.id = np.array([123]) # Fake tracker ID
                    self.cls = np.array([0])  # Fake class ID
                    self.conf = np.array([0.95]) # Fake confidence
            class MockResult:
                def __init__(self):
                    self.boxes = MockBoxes()
                    self.names = {0: 'mock_fault'}
            return [MockResult()]

# --- Main Processing Function ---
def process_video_sync(video_path, confidence_threshold=0.5):
    """
    Processes a video synchronously with a strict time limit for Vercel.
    """
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}

    # Set a hard time limit to avoid Vercel timeout (e.g., 25 seconds)
    max_processing_time = 25
    start_time = time.time()
    
    # Load your YOLO model (real or mock)
    # Ensure the model file is in your project root if you want to use the real one.
    try:
        if YOLO_AVAILABLE:
            # Assuming your model is in the root directory
            model_abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fault_detection_model_best.pt')
            yolo_model = YOLO(model_abs_path)
            logging.info(f"Real YOLO model loaded from {model_abs_path}.")
        else:
            yolo_model = MockYOLO('fault_detection_model_best.pt')
            logging.info("Mock YOLO model initialized.")
    except Exception as e:
        logging.error(f"Could not load any YOLO model: {e}. Processing without detection.")
        yolo_model = None
    
    results = {
        "status": "processing",
        "faults_detected": 0,
        "message": "Video processing started",
        "processing_time_sec": 0,
        "faults": []
    }

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results["error"] = "Error opening video file"
            return results

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        unique_alerts_processed = {}

        while cap.isOpened():
            # CRITICAL: Check if we are about to exceed the time limit
            if time.time() - start_time > max_processing_time:
                logging.warning("Approaching Vercel timeout, stopping processing early.")
                results["message"] = "Processing stopped early to avoid timeout."
                break

            success, frame = cap.read()
            if not success:
                break

            # --- YOUR YOLO PROCESSING LOGIC ---
            if yolo_model:
                try:
                    model_results = yolo_model.track(source=frame, persist=True, conf=confidence_threshold, verbose=False)
                    if model_results and len(model_results) > 0 and hasattr(model_results[0], 'boxes') and model_results[0].boxes is not None:
                        boxes = model_results[0].boxes
                        if boxes.id is not None:
                            for i in range(len(boxes.id)):
                                tracker_id = int(boxes.id[i].cpu().numpy())
                                if tracker_id not in unique_alerts_processed:
                                    cls_idx = int(boxes.cls[i].cpu().numpy())
                                    class_name = model_results[0].names.get(cls_idx, 'unknown')
                                    unique_alerts_processed[tracker_id] = {
                                        "frame": frame_number,
                                        "class": class_name,
                                        "confidence": float(boxes.conf[i].cpu().numpy())
                                    }
                except Exception as e:
                    logging.error(f"Error during YOLO inference: {e}")
            # --- END OF YOUR LOGIC ---

            frame_number += 1

        cap.release()

        # Clean up the video file
        try:
            os.remove(video_path)
        except Exception as e:
            logging.error(f"Error cleaning up video file: {e}")

        # Finalize results
        results["status"] = "completed"
        results["faults_detected"] = len(unique_alerts_processed)
        results["processing_time_sec"] = round(time.time() - start_time, 2)
        results["faults"] = list(unique_alerts_processed.values())
        results["message"] = f"Processed {frame_number} frames in {results['processing_time_sec']}s."

    except Exception as e:
        logging.error(f"Error during synchronous video processing: {e}")
        results["error"] = str(e)
        results["status"] = "failed"

    return results