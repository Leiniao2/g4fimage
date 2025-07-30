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
    
    # Method 1: Try g4f library with image generation
    try:
        import g4f
        from g4f.Provider import Bing, OpenaiChat, You
        
        # Try multiple providers that support image generation
        providers = [Bing, You, OpenaiChat]
        
        for provider in providers:
            try:
                logging.info(f"Trying provider: {provider.__name__}")
                
                # For image generation, we need to use a model that supports it
                response = g4f.ChatCompletion.create(
                    model=g4f.models.gpt_4,
                    messages=[{
                        "role": "user", 
                        "content": f"Create an image of: {prompt}"
                    }],
                    provider=provider,
                )
                
                if response and len(response) > 10:  # Basic check for valid response
                    logging.info(f"Success with provider: {provider.__name__}")
                    return {
                        'success': True,
                        'image_data': response,
                        'provider': provider.__name__
                    }
                    
            except Exception as provider_error:
                logging.warning(f"Provider {provider.__name__} failed: {str(provider_error)}")
                continue
                
    except ImportError:
        logging.error("g4f library not available")
    except Exception as e:
        logging.error(f"g4f general error: {str(e)}")
    
    # Method 2: Try alternative g4f approach with different models
    try:
        import g4f
        
        # Try with image-specific prompt
        image_prompt = f"Generate a detailed image of: {prompt}. Make it high quality and artistic."
        
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": image_prompt}],
            stream=False,
        )
        
        if response:
            return {
                'success': True,
                'image_data': response,
                'method': 'alternative_g4f'
            }
            
    except Exception as e:
        logging.error(f"Alternative g4f method failed: {str(e)}")
    
    # Method 3: Try using Hugging Face API (free tier)
    try:
        hf_response = generate_with_huggingface(prompt)
        if hf_response.get('success'):
            return hf_response
    except Exception as e:
        logging.error(f"Hugging Face method failed: {str(e)}")
    
    # Method 4: Mock/Demo response for testing
    if os.environ.get('FLASK_ENV') == 'development':
        return generate_demo_image(prompt)
    
    return {'success': False, 'error': 'All image generation methods failed'}

def generate_with_huggingface(prompt):
    """Try generating image using Hugging Face Inference API"""
    try:
        # Using Stable Diffusion via Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        
        # You can get a free API token from huggingface.co
        headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_TOKEN', '')}"
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
            logging.error(f"Hugging Face API error: {response.status_code}")
            
    except Exception as e:
        logging.error(f"Hugging Face error: {str(e)}")
    
    return {'success': False}

def generate_demo_image(prompt):
    """Generate a demo placeholder image for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        
        # Create a simple placeholder image
        width, height = 512, 512
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            # Try to use default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Wrap text
        lines = []
        words = prompt.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) < 25:  # Simple wrap at 25 chars
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw text lines
        y_start = height // 2 - (len(lines) * 20) // 2
        for i, line in enumerate(lines[:5]):  # Max 5 lines
            draw.text((50, y_start + i * 25), line, fill='darkblue', font=font)
        
        # Add demo watermark
        draw.text((10, height - 30), "DEMO IMAGE", fill='red', font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'success': True,
            'image_data': f"data:image/png;base64,{img_base64}",
            'method': 'demo',
            'note': 'This is a demo placeholder. Configure API keys for real generation.'
        }
        
    except Exception as e:
        logging.error(f"Demo image generation failed: {str(e)}")
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
