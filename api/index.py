import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the app and the patching function
try:
    from app import app as flask_app
    from api import patch_app_for_vercel
    
    # Apply the patches
    patch_app_for_vercel(flask_app)
    logger.info("App patched successfully")
except Exception as e:
    logger.error(f"Error importing or patching app: {e}")
    flask_app = None

# Vercel serverless function handler
def handler(environ, start_response):
    if flask_app is None:
        start_response('500 Internal Server Error', [('Content-type', 'text/plain')])
        return [b"Server initialization error"]
    
    try:
        return flask_app(environ, start_response)
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        start_response('500 Internal Server Error', [('Content-type', 'text/plain')])
        return [f"Error: {str(e)}".encode()]

# For Vercel's Python runtime compatibility
if flask_app:
    app_handler = flask_app.wsgi_app