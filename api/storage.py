import os
import base64
import logging
from vercel_blob import put, del_, list

class VercelBlobStorage:
    def __init__(self):
        self.token = os.environ.get('BLOB_READ_WRITE_TOKEN')
        if not self.token:
            logging.warning("BLOB_READ_WRITE_TOKEN not set. File storage will be limited.")
    
    async def save_file(self, file_data, filename):
        if not self.token:
            # Fallback to base64 encoding for small files
            if len(file_data) < 1000000:  # 1MB limit
                return base64.b64encode(file_data).decode('utf-8')
            return None
        
        try:
            blob = await put(filename, file_data, {
                'access': 'public',
                'token': self.token
            })
            return blob.url
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            return None
    
    async def delete_file(self, filename):
        if not self.token:
            return False
        
        try:
            await del_(filename, {'token': self.token})
            return True
        except Exception as e:
            logging.error(f"Failed to delete file: {e}")
            return False

# Create a global storage instance
storage = VercelBlobStorage()