# src/index.py
import os
import json
import base64
from io import BytesIO
import time
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.id import ID

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.utils.helper import handle_overlapping_boxes, group_aligned_labels, cleanning_text, cleanning_num

def main(context):
    # Parse request data
    request_data = context.req.body
    
    # Initialize response
    response = {
        'success': False,
        'message': '',
        'data': None
    }
    
    try:
        # Check if request is multipart/form-data
        if context.req.headers.get('content-type', '').startswith('multipart/form-data'):
            # Get file from request
            file_data = context.req.files.get('image')
            if not file_data:
                response['message'] = 'No image file provided'
                return context.res.json(response, 400)
            
            # Save file to temporary directory
            filename = file_data.filename
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file_data.save(file_path)
        
        # Check if request is JSON with base64 image
        elif context.req.headers.get('content-type', '') == 'application/json':
            data = request_data
            if not data or 'image' not in data:
                response['message'] = 'No image data provided'
                return context.res.json(response, 400)
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image'])
                filename = data.get('filename', f"image_{int(time.time())}.jpg")
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                with open(file_path, 'wb') as f:
                    f.write(image_data)
            except Exception as e:
                response['message'] = f'Invalid base64 image: {str(e)}'
                return context.res.json(response, 400)
        else:
            response['message'] = 'Unsupported content type'
            return context.res.json(response, 400)
        
        # Initialize ImageProcessingService
        from app.image_processing_service import ImageProcessingService
        service = ImageProcessingService()
        
        # Process image
        result_json_path, processed_image_path, extracted_text = service.process_image(file_path, filename)
        
        # Read results
        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
        
        # Read processed image
        with open(processed_image_path, 'rb') as f:
            processed_image_bytes = f.read()
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        
        # Upload results to Appwrite Storage if environment variables are set
        storage_urls = {}
        if os.environ.get('APPWRITE_ENDPOINT') and os.environ.get('APPWRITE_API_KEY') and os.environ.get('APPWRITE_BUCKET_ID'):
            try:
                client = Client()
                client.set_endpoint(os.environ.get('APPWRITE_ENDPOINT'))
                client.set_project(os.environ.get('APPWRITE_PROJECT_ID'))
                client.set_key(os.environ.get('APPWRITE_API_KEY'))
                
                storage = Storage(client)
                
                # Upload processed image
                image_file_id = ID.unique()
                storage.create_file(
                    bucket_id=os.environ.get('APPWRITE_BUCKET_ID'),
                    file_id=image_file_id,
                    file=open(processed_image_path, 'rb')
                )
                storage_urls['processed_image'] = f"{os.environ.get('APPWRITE_ENDPOINT')}/storage/buckets/{os.environ.get('APPWRITE_BUCKET_ID')}/files/{image_file_id}/view"
                
                # Upload JSON result
                json_file_id = ID.unique()
                storage.create_file(
                    bucket_id=os.environ.get('APPWRITE_BUCKET_ID'),
                    file_id=json_file_id,
                    file=open(result_json_path, 'rb')
                )
                storage_urls['result_json'] = f"{os.environ.get('APPWRITE_ENDPOINT')}/storage/buckets/{os.environ.get('APPWRITE_BUCKET_ID')}/files/{json_file_id}/view"
            except Exception as e:
                # Just log the error, don't fail the function
                print(f"Error uploading to Appwrite Storage: {str(e)}")
        
        # Prepare response
        response['success'] = True
        response['message'] = 'Image processed successfully'
        response['data'] = {
            'result': result_json,
            'processed_image': processed_image_base64,
            'extracted_text': extracted_text,
            'storage_urls': storage_urls
        }
        
        return context.res.json(response)
        
    except Exception as e:
        import traceback
        response['message'] = f'Error processing image: {str(e)}'
        response['error_details'] = traceback.format_exc()
        return context.res.json(response, 500)
    finally:
        # Clean up temporary files
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        if 'result_json_path' in locals() and os.path.exists(result_json_path):
            os.remove(result_json_path)
        if 'processed_image_path' in locals() and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
