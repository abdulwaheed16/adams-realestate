# api/index.py
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import patches and initialization
from api import init_app

# Initialize the app with Vercel-specific patches
app = init_app()

# Vercel serverless function handler
def handler(request):
    return app(request.environ, start_response)

# For Vercel's Python runtime
app_handler = app.as_wsgi_app = app.wsgi_app