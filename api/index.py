import sys
import os

# Add the project root to the Python path so we can import app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app and the patching function
from app import app as flask_app
from api import patch_app_for_vercel

# Apply the patches to modify the app's behavior for Vercel
patch_app_for_vercel(flask_app)

# This is the standard Vercel serverless function handler
def handler(environ, start_response):
    return flask_app(environ, start_response)

# For Vercel's Python runtime compatibility
app_handler = flask_app.wsgi_app