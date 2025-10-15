import os
import tempfile
import shutil
import logging

def patch_app_for_vercel(app_instance):
    """
    Patches the Flask app instance to be compatible with Vercel's serverless environment.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Patching Flask app for Vercel deployment...")

    # 1. Replace permanent folders with a temporary directory
    temp_dir = tempfile.mkdtemp()
    app_instance.config['UPLOAD_FOLDER'] = temp_dir
    app_instance.config['CLIPS_FOLDER'] = temp_dir
    logging.info(f"Using temporary directory: {temp_dir}")

    # 2. Replace the background processing with a synchronous version
    from api.video_processor import process_video_sync
    
    # Find the original upload_file route and replace it
    original_upload = app_instance.view_functions.get('upload_file')
    if original_upload:
        def new_upload_file():
            from flask import request, jsonify
            if 'video' not in request.files:
                return jsonify({"error": "No video file provided"}), 400
            
            file = request.files['video']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            video_path = os.path.join(app_instance.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Call the NEW synchronous processor instead of starting a thread
            result = process_video_sync(video_path, confidence_threshold=0.5)
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

            return jsonify(result)
        
        # Replace the original view function with our new one
        app_instance.view_functions['upload_file'] = new_upload_file
        logging.info("Replaced 'upload_file' with a synchronous, Vercel-compatible version.")

    logging.info("Patching complete.")