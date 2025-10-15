import os
import time
import logging
import cv2
import numpy as np

# This function is a synchronous, time-limited version of your background processor
def process_video_sync(video_path, confidence_threshold=0.5):
    """
    Processes a video synchronously with a strict time limit for Vercel.
    """
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}

    # Set a hard time limit to avoid Vercel timeout (e.g., 25 seconds)
    max_processing_time = 25
    start_time = time.time()
    
    # Load your YOLO model. Ensure the model file is in your project directory.
    # For this example, we'll use a mock to avoid dependency issues.
    # In your real code, you would load it as you did in app.py
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt') # Replace with your model path if needed
        logging.info("YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"Could not load YOLO model: {e}. Processing without detection.")
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

            # --- YOUR YOLO PROCESSING LOGIC GOES HERE ---
            if yolo_model:
                try:
                    model_results = yolo_model.track(source=frame, persist=True, conf=confidence_threshold, verbose=False)
                    # ... process model_results to find faults and populate unique_alerts_processed ...
                    # This is a placeholder for your actual logic.
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