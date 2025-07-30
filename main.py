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
    """Generate image using selected provider"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        provider = data.get('provider', 'auto')  # Default to auto-select
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        logging.info(f"Generating image for prompt: '{prompt}' using provider: {provider}")
        
        # Try to generate image with selected provider
        if provider == 'auto':
            response = call_gpt4free_api(prompt)  # Use auto-selection logic
        else:
            response = generate_with_specific_provider(prompt, provider)
        
        if response.get('success'):
            result = {
                'success': True,
                'image_data': response.get('image_data'),
                'image_url': response.get('image_url'),
                'method': response.get('method', provider),
                'provider': response.get('provider', provider),
                'generation_time': response.get('generation_time', 'N/A')
            }
            
            # Add note if it's a demo
            if response.get('note'):
                result['note'] = response.get('note')
            
            logging.info(f"Image generated successfully using: {result.get('method')}")
            return jsonify(result)
        else:
            error_msg = response.get('error', f'Failed to generate image with {provider}')
            logging.error(f"Image generation failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': response.get('details', 'Try selecting a different provider or check your API configuration')
            }), 500
            
    except Exception as e:
        logging.error(f"Error in generate_image endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if os.environ.get('FLASK_ENV') == 'development' else 'Please try again'
        }), 500

@app.route('/providers')
def get_providers():
    """Get list of available providers with their status"""
    providers = {
        'pollinations': {
            'name': 'Pollinations.ai',
            'description': 'Free AI image generation, no API key required',
            'status': 'available',
            'free': True,
            'quality': 'Good',
            'speed': 'Fast (5-15s)',
            'requirements': 'None'
        },
        'huggingface': {
            'name': 'Hugging Face (Stable Diffusion)',
            'description': 'High-quality Stable Diffusion models',
            'status': 'available' if os.environ.get('HUGGINGFACE_API_TOKEN') else 'needs_token',
            'free': True,
            'quality': 'Excellent',
            'speed': 'Medium (10-30s)',
            'requirements': 'Free API token from huggingface.co'
        },
        'deepai': {
            'name': 'DeepAI',
            'description': 'Professional AI image generation',
            'status': 'available' if os.environ.get('DEEPAI_API_KEY') else 'needs_token',
            'free': False,
            'quality': 'Very Good',
            'speed': 'Fast (5-20s)',
            'requirements': 'API key from deepai.org (has free tier)'
        },
        'replicate': {
            'name': 'Replicate',
            'description': 'Multiple AI models including DALL-E style',
            'status': 'available' if os.environ.get('REPLICATE_API_TOKEN') else 'needs_token',
            'free': False,
            'quality': 'Excellent',
            'speed': 'Medium (15-45s)',
            'requirements': 'API token from replicate.com'
        },
        'stability': {
            'name': 'Stability AI',
            'description': 'Official Stable Diffusion API',
            'status': 'available' if os.environ.get('STABILITY_API_KEY') else 'needs_token',
            'free': False,
            'quality': 'Excellent',
            'speed': 'Fast (10-25s)',
            'requirements': 'API key from stability.ai'
        },
        'openai': {
            'name': 'OpenAI DALL-E',
            'description': 'High-quality DALL-E image generation',
            'status': 'available' if os.environ.get('OPENAI_API_KEY') else 'needs_token',
            'free': False,
            'quality': 'Excellent',
            'speed': 'Fast (10-30s)',
            'requirements': 'API key from openai.com'
        },
        'demo': {
            'name': 'Demo Mode',
            'description': 'Colorful placeholder images for testing',
            'status': 'available',
            'free': True,
            'quality': 'Demo Only',
            'speed': 'Instant',
            'requirements': 'None'
        }
    }
    
    return jsonify(providers)

def generate_with_specific_provider(prompt, provider):
    """Generate image using a specific provider"""
    import time
    start_time = time.time()
    
    try:
        if provider == 'pollinations':
            result = generate_with_pollinations(prompt)
        elif provider == 'huggingface':
            result = generate_with_huggingface(prompt)
        elif provider == 'deepai':
            result = generate_with_deepai(prompt)
        elif provider == 'replicate':
            result = generate_with_replicate(prompt)
        elif provider == 'stability':
            result = generate_with_stability(prompt)
        elif provider == 'openai':
            result = generate_with_openai(prompt)
        elif provider == 'demo':
            result = generate_demo_image(prompt)
        else:
            return {'success': False, 'error': f'Unknown provider: {provider}'}
        
        # Add generation time
        if result.get('success'):
            result['generation_time'] = f"{time.time() - start_time:.1f}s"
        
        return result
        
    except Exception as e:
        logging.error(f"Error with provider {provider}: {str(e)}")
        return {'success': False, 'error': f'Provider {provider} failed: {str(e)}'}

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

def generate_with_replicate(prompt):
    """Generate image using Replicate API"""
    try:
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not api_token:
            return {'success': False, 'error': 'Replicate API token not configured'}
        
        headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        }
        
        # Using SDXL model on Replicate
        data = {
            "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "input": {
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "num_outputs": 1,
                "scheduler": "K_EULER",
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        
        response = requests.post(
            'https://api.replicate.com/v1/predictions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 201:
            prediction = response.json()
            prediction_id = prediction['id']
            
            # Poll for completion
            for _ in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                status_response = requests.get(
                    f'https://api.replicate.com/v1/predictions/{prediction_id}',
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    result = status_response.json()
                    if result['status'] == 'succeeded' and result['output']:
                        # Download image and convert to base64
                        img_url = result['output'][0]
                        img_response = requests.get(img_url, timeout=30)
                        
                        if img_response.status_code == 200:
                            import base64
                            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                            
                            return {
                                'success': True,
                                'image_data': f"data:image/png;base64,{image_base64}",
                                'method': 'replicate'
                            }
                    elif result['status'] == 'failed':
                        return {'success': False, 'error': 'Replicate generation failed'}
        
    except Exception as e:
        logging.error(f"Replicate error: {str(e)}")
    
    return {'success': False, 'error': 'Replicate API failed'}

def generate_with_stability(prompt):
    """Generate image using Stability AI API"""
    try:
        api_key = os.environ.get('STABILITY_API_KEY')
        if not api_key:
            return {'success': False, 'error': 'Stability AI API key not configured'}
        
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 20,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            for i, image in enumerate(data["artifacts"]):
                return {
                    'success': True,
                    'image_data': f"data:image/png;base64,{image['base64']}",
                    'method': 'stability'
                }
        else:
            logging.error(f"Stability AI error: {response.status_code} - {response.text}")
    
    except Exception as e:
        logging.error(f"Stability AI error: {str(e)}")
    
    return {'success': False, 'error': 'Stability AI API failed'}

def generate_with_openai(prompt):
    """Generate image using OpenAI DALL-E API"""
    try:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {'success': False, 'error': 'OpenAI API key not configured'}
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'dall-e-3',
            'prompt': prompt,
            'n': 1,
            'size': '1024x1024',
            'response_format': 'b64_json'
        }
        
        response = requests.post(
            'https://api.openai.com/v1/images/generations',
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['data'] and len(result['data']) > 0:
                image_data = result['data'][0]['b64_json']
                return {
                    'success': True,
                    'image_data': f"data:image/png;base64,{image_data}",
                    'method': 'openai'
                }
        else:
            logging.error(f"OpenAI API error: {response.status_code} - {response.text}")
    
    except Exception as e:
        logging.error(f"OpenAI error: {str(e)}")
    
    return {'success': False, 'error': 'OpenAI API failed'}
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

@app.route('/test-image')
def test_image():
    """Test endpoint to verify image generation works"""
    try:
        result = generate_demo_image("Test image - beautiful sunset over mountains")
        
        if result.get('success'):
            return f'''
            <html>
                <body>
                    <h2>Image Generation Test</h2>
                    <p>Method: {result.get('method')}</p>
                    <p>Note: {result.get('note', 'N/A')}</p>
                    <img src="{result.get('image_data')}" style="max-width: 500px;">
                    <br><br>
                    <a href="/">Back to main app</a>
                </body>
            </html>
            '''
        else:
            return f"<h2>Test Failed</h2><p>Error: {result.get('error')}</p><a href='/'>Back</a>"
            
    except Exception as e:
        return f"<h2>Test Exception</h2><p>{str(e)}</p><a href='/'>Back</a>"

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
