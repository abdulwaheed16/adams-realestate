# api/__init__.py
import os
import sys
import tempfile
import shutil
from werkzeug.utils import secure_filename

# Create a global file handler instance
class VercelFileHandler:
    def __init__(self):
        # Use a temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def save_file(self, file, filename=None):
        if filename is None:
            filename = secure_filename(file.filename)
        
        filepath = os.path.join(self.temp_dir, filename)
        file.save(filepath)
        return filepath
    
    def delete_file(self, file_path):
        try:
            os.remove(file_path)
            return True
        except:
            return False
    
    def get_file_path(self, filename):
        return os.path.join(self.temp_dir, filename)
    
    def cleanup(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir, ignore_errors=True)

file_handler = VercelFileHandler()

# Monkey-patch file operations in your app
def init_app():
    # Import your app module
    import app
    
    # Override app config for Vercel
    app.app.config['UPLOAD_FOLDER'] = file_handler.temp_dir
    app.app.config['CLIPS_FOLDER'] = file_handler.temp_dir
    
    # Replace background processing with synchronous processing
    from api.video_processor import process_video_sync
    app.process_video_in_background = process_video_sync
    
    # Return the patched app
    return app.app