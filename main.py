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
    """Generate image using multiple fallback methods"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        logging.info(f"Generating image for prompt: {prompt}")
        
        # Try to generate image
        response = call_gpt4free_api(prompt)
        
        if response.get('success'):
            result = {
                'success': True,
                'image_data': response.get('image_data'),
                'image_url': response.get('image_url'),
                'method': response.get('method', 'unknown'),
                'provider': response.get('provider', 'unknown')
            }
            
            # Add note if it's a demo
            if response.get('note'):
                result['note'] = response.get('note')
            
            logging.info(f"Image generated successfully using: {result.get('method')}")
            return jsonify(result)
        else:
            error_msg = response.get('error', 'Failed to generate image with all methods')
            logging.error(f"Image generation failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': 'Try checking your API configuration or network connection'
            }), 500
            
    except Exception as e:
        logging.error(f"Error in generate_image endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if os.environ.get('FLASK_ENV') == 'development' else 'Please try again'
        }), 500

def call_gpt4free_api(prompt):
    """Call the gpt4free API to generate an image using multiple methods"""
    
    # Method 1: Try using Hugging Face API first (most reliable)
    try:
        logging.info("Trying Hugging Face API...")
        hf_response = generate_with_huggingface(prompt)
        if hf_response.get('success'):
            return hf_response
    except Exception as e:
        logging.error(f"Hugging Face method failed: {str(e)}")
    
    # Method 2: Try Pollinations API (free, no auth required)
    try:
        logging.info("Trying Pollinations API...")
        pollinations_response = generate_with_pollinations(prompt)
        if pollinations_response.get('success'):
            return pollinations_response
    except Exception as e:
        logging.error(f"Pollinations method failed: {str(e)}")
    
    # Method 3: Try DeepAI API (has free tier)
    try:
        logging.info("Trying DeepAI API...")
        deepai_response = generate_with_deepai(prompt)
        if deepai_response.get('success'):
            return deepai_response
    except Exception as e:
        logging.error(f"DeepAI method failed: {str(e)}")
    
    # Method 4: Always provide demo image as final fallback
    logging.info("All APIs failed, generating demo image...")
    return generate_demo_image(prompt)

def generate_with_pollinations(prompt):
    """Generate image using Pollinations API (free, no auth required)"""
    try:
        # Pollinations.ai provides free image generation
        encoded_prompt = requests.utils.quote(prompt)
        
        # Try multiple endpoints
        endpoints = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true",
            f"https://pollinations.ai/p/{encoded_prompt}?width=512&height=512"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=30)
                
                if response.status_code == 200:
                    # Convert image bytes to base64
                    import base64
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    
                    return {
                        'success': True,
                        'image_data': f"data:image/png;base64,{image_base64}",
                        'method': 'pollinations'
                    }
            except Exception as e:
                logging.warning(f"Pollinations endpoint failed: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Pollinations error: {str(e)}")
    
    return {'success': False}

def generate_with_deepai(prompt):
    """Generate image using DeepAI API"""
    try:
        # DeepAI has a free tier - you can get API key from deepai.org
        api_key = os.environ.get('DEEPAI_API_KEY')
        if not api_key:
            return {'success': False}
            
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            data={
                'text': prompt,
            },
            headers={'api-key': api_key},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'output_url' in result:
                # Download the image and convert to base64
                img_response = requests.get(result['output_url'], timeout=30)
                if img_response.status_code == 200:
                    import base64
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    
                    return {
                        'success': True,
                        'image_data': f"data:image/png;base64,{image_base64}",
                        'method': 'deepai'
                    }
        
    except Exception as e:
        logging.error(f"DeepAI error: {str(e)}")
    
    return {'success': False}

def generate_with_huggingface(prompt):
    """Generate image using Hugging Face Inference API"""
    try:
        api_token = os.environ.get('HUGGINGFACE_API_TOKEN')
        if not api_token:
            logging.info("No Hugging Face API token found")
            return {'success': False}
            
        # Using Stable Diffusion via Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        
        headers = {
            "Authorization": f"Bearer {api_token}"
        }
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200:
            # Convert image bytes to base64
            import base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_base64}",
                'method': 'huggingface'
            }
        else:
            logging.error(f"Hugging Face API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logging.error(f"Hugging Face error: {str(e)}")
    
    return {'success': False}

def generate_demo_image(prompt):
    """Generate a demo placeholder image for testing - GUARANTEED TO WORK"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        
        # Create a colorful placeholder image
        width, height = 512, 512
        
        # Create gradient background
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for i in range(width):
            for j in range(height):
                # Create a nice gradient
                r = int(255 * (i / width))
                g = int(255 * (j / height))
                b = int(255 * ((i + j) / (width + height)))
                pixels[i, j] = (r % 255, g % 255, b % 255)
        
        draw = ImageDraw.Draw(img)
        
        # Add border
        draw.rectangle([10, 10, width-10, height-10], outline='white', width=3)
        
        # Add title
        draw.rectangle([20, 20, width-20, 80], fill='black', outline='white', width=2)
        draw.text((30, 35), "AI Generated Image", fill='white')
        
        # Wrap and draw prompt text
        words = prompt.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) < 30:  # Wrap at 30 chars
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw prompt text with background
        text_height = len(lines) * 25 + 40
        draw.rectangle([40, 120, width-40, 120 + text_height], fill='rgba(0,0,0,128)')
        
        y_start = 140
        for i, line in enumerate(lines[:8]):  # Max 8 lines
            draw.text((50, y_start + i * 25), line, fill='white')
        
        # Add demo watermark
        draw.text((20, height - 60), "DEMO MODE", fill='yellow')
        draw.text((20, height - 35), "Configure APIs for real generation", fill='yellow')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logging.info("Demo image generated successfully")
        
        return {
            'success': True,
            'image_data': f"data:image/png;base64,{img_base64}",
            'method': 'demo',
            'note': 'This is a demo placeholder. Add API keys for real AI generation.'
        }
        
    except Exception as e:
        logging.error(f"Demo image generation failed: {str(e)}")
        
        # Ultimate fallback - create minimal image without PIL
        try:
            # Create a simple 1x1 pixel image as absolute fallback
            import base64
            # This is a 1x1 red pixel PNG in base64
            red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{red_pixel}",
                'method': 'minimal_fallback',
                'note': 'Minimal fallback image - check your Python environment'
            }
        except:
            return {'success': False, 'error': 'All fallback methods failed'}

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
