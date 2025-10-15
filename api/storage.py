import os
import logging
from vercel_blob import put

logger = logging.getLogger(__name__)

def upload_to_blob(file_path, filename):
    """Upload a file to Vercel Blob storage"""
    try:
        token = os.environ.get('BLOB_READ_WRITE_TOKEN')
        if not token:
            logger.error("BLOB_READ_WRITE_TOKEN not set")
            return None
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        blob = put(filename, file_data, {
            'access': 'public',
            'token': token
        })
        
        logger.info(f"File uploaded to blob: {blob.url}")
        return blob.url
        
    except Exception as e:
        logger.error(f"Error uploading to blob: {e}")
        return None