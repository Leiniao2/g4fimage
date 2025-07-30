from flask import Flask, request, jsonify, render_template
import requests
import base64
import io
from PIL import Image
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate image using gpt4free API"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Example using g4f (you may need to adjust based on actual gpt4free API)
        # This is a mock implementation - adjust based on the actual gpt4free API structure
        response = call_gpt4free_api(prompt)
        
        if response.get('success'):
            return jsonify({
                'success': True,
                'image_url': response.get('image_url'),
                'image_data': response.get('image_data')
            })
        else:
            return jsonify({'error': 'Failed to generate image'}), 500
            
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def call_gpt4free_api(prompt):
    """Call the gpt4free API to generate an image"""
    try:
        # Method 1: Using direct API call (adjust URL based on actual endpoint)
        api_url = "https://api.gpt4free.io/generate-image"  # Example URL
        
        payload = {
            "prompt": prompt,
            "model": "dalle-3",  # or whatever model is available
            "size": "1024x1024"
        }
        
        headers = {
            "Content-Type": "application/json",
            # Add any required headers/API keys here
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'image_url': result.get('image_url'),
                'image_data': result.get('image_data')
            }
        else:
            logging.error(f"API call failed: {response.status_code}")
            return {'success': False}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {str(e)}")
        
        # Method 2: Fallback using g4f library (if available)
        try:
            import g4f
            
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Generate an image: {prompt}"}],
                provider=g4f.Provider.Bing,  # or another provider that supports images
            )
            
            # Process the response to extract image data
            # This will depend on how g4f returns image data
            if response:
                return {
                    'success': True,
                    'image_data': response  # Adjust based on actual response format
                }
            
        except ImportError:
            logging.error("g4f library not available")
        except Exception as e:
            logging.error(f"g4f error: {str(e)}")
        
        return {'success': False}

@app.route('/proxy-image')
def proxy_image():
    """Proxy image requests to handle CORS issues"""
    image_url = request.args.get('url')
    
    if not image_url:
        return jsonify({'error': 'URL parameter required'}), 400
    
    try:
        response = requests.get(image_url, timeout=10)
        
        if response.status_code == 200:
            # Return the image with proper headers
            return response.content, 200, {
                'Content-Type': response.headers.get('Content-Type', 'image/png'),
                'Access-Control-Allow-Origin': '*'
            }
        else:
            return jsonify({'error': 'Failed to fetch image'}), 404
            
    except Exception as e:
        logging.error(f"Error proxying image: {str(e)}")
        return jsonify({'error': 'Failed to proxy image'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
