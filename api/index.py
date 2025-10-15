import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Flask app
try:
    from app import app as flask_app
    logger.info("Flask app imported successfully")
except Exception as e:
    logger.error(f"Error importing Flask app: {e}")
    flask_app = None

# Vercel serverless function handler
def handler(environ, start_response):
    """Main handler for Vercel serverless functions"""
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