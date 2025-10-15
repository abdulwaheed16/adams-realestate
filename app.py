import os
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing this file
basedir = os.path.abspath(os.path.dirname(__file__))

# Create Flask app with correct template and static folders
app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

# App configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'temp')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Ensure temp directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')
VIDEO_PROCESSING_SERVICE_URL = os.environ.get('VIDEO_PROCESSING_SERVICE_URL', '')
BLOB_READ_WRITE_TOKEN = os.environ.get('BLOB_READ_WRITE_TOKEN', '')

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set")

if not N8N_WEBHOOK_URL:
    logger.warning("N8N_WEBHOOK_URL not set")

# Routes
@app.route('/', methods=['GET'])
def index():
    """Render the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return jsonify({"error": "Template not found"}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video upload and trigger processing"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Check file extension
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
        filename = secure_filename(file.filename)
        if not '.' in filename or filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Upload to Vercel Blob
        from api.storage import upload_to_blob
        blob_url = upload_to_blob(temp_path, filename)
        
        if not blob_url:
            return jsonify({"error": "Failed to upload file to storage"}), 500
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Trigger processing service
        if VIDEO_PROCESSING_SERVICE_URL:
            processing_response = trigger_processing_service(blob_url, filename)
            if processing_response:
                return jsonify({
                    "status": "processing",
                    "message": "Video uploaded and processing started",
                    "job_id": processing_response.get('job_id'),
                    "video_url": blob_url
                })
        
        # Fallback: return success without processing
        return jsonify({
            "status": "uploaded",
            "message": "Video uploaded successfully",
            "video_url": blob_url
        })
        
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/processing-complete', methods=['POST'])
def processing_complete():
    """Webhook endpoint for processing service to send results"""
    try:
        results = request.json
        logger.info(f"Received processing results: {results}")
        
        # Forward results to N8N webhook
        if N8N_WEBHOOK_URL:
            import requests
            requests.post(N8N_WEBHOOK_URL, json=results, timeout=10)
        
        return jsonify({"status": "received"}), 200
        
    except Exception as e:
        logger.error(f"Error in processing_complete webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "services": {
            "openai": bool(OPENAI_API_KEY),
            "n8n_webhook": bool(N8N_WEBHOOK_URL),
            "blob_storage": bool(BLOB_READ_WRITE_TOKEN),
            "processing_service": bool(VIDEO_PROCESSING_SERVICE_URL)
        }
    })

def trigger_processing_service(video_url, filename):
    """Trigger external video processing service"""
    try:
        import requests
        
        payload = {
            "video_url": video_url,
            "filename": filename,
            "callback_url": f"{request.url_root}webhook/processing-complete",
            "openai_api_key": OPENAI_API_KEY
        }
        
        response = requests.post(
            f"{VIDEO_PROCESSING_SERVICE_URL}/process",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Processing service error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error triggering processing service: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))