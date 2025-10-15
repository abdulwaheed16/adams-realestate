import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup():
    """Setup script that runs after dependencies are installed"""
    logger.info("Running post-install setup...")
    
    # Create necessary directories
    directories = ['static', 'templates', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create a basic template if it doesn't exist
    template_path = Path('templates/index.html')
    if not template_path.exists():
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Service</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
        .status { margin-top: 20px; padding: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Processing Service</h1>
        <p>Upload a video for fault detection and analysis</p>
        
        <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required>
            <br><br>
            <button type="submit">Upload and Process</button>
        </form>
        
        <div id="status"></div>
    </div>
    
    <script>
        document.querySelector('form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const statusDiv = document.getElementById('status');
            
            statusDiv.innerHTML = '<div class="status">Uploading and processing...</div>';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    statusDiv.innerHTML = `<div class="status success">${result.message}</div>`;
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${result.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Network error: ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
        """
        template_path.write_text(template_content)
        logger.info("Created default template")
    
    logger.info("Setup complete!")

if __name__ == '__main__':
    setup()