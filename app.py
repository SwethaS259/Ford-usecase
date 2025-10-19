# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from utils import load_whisper_model, transcribe_audio, transcribe_audio_simple, allowed_image_file, save_uploaded_file, save_base64_audio, convert_audio_to_wav, check_ffmpeg_available, TORCH_AVAILABLE, HUGGINGFACE_TOKEN
import os
import uuid
import time
import traceback
import base64
import numpy as np

app = Flask(__name__)
app.secret_key = 'voice-image-metadata-secret-key-2023'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Global variables for model
processor, model = None, None

def initialize_model():
    """Initialize Whisper model on startup"""
    global processor, model
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping model initialization")
        return
        
    try:
        print("Initializing Whisper model...")
        processor, model = load_whisper_model()
        if processor is None or model is None:
            print("Warning: Whisper model failed to load. Audio transcription will not be available.")
        else:
            print("✓ Whisper model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(traceback.format_exc())

@app.route('/')
def index():
    """Main page with upload form"""
    model_status = "loaded" if (processor and model) else "not loaded"
    torch_status = "available" if TORCH_AVAILABLE else "not available"
    token_status = "available" if HUGGINGFACE_TOKEN else "not set"
    ffmpeg_status = "available" if check_ffmpeg_available() else "not available"
    
    # Language options for the template
    languages = [
        ('auto', 'Auto-detect Language'),
        ('english', 'English'),
        ('spanish', 'Spanish'),
        ('french', 'French'),
        ('german', 'German'),
        ('chinese', 'Chinese'),
        ('japanese', 'Japanese'),
        ('korean', 'Korean'),
        ('hindi', 'Hindi'),
        ('arabic', 'Arabic'),
        ('russian', 'Russian'),
        ('portuguese', 'Portuguese'),
        ('italian', 'Italian')
    ]
    
    return render_template('index.html', 
                         model_status=model_status, 
                         torch_status=torch_status,
                         token_status=token_status,
                         ffmpeg_status=ffmpeg_status,
                         languages=languages)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and process audio transcription"""
    try:
        # Check if images were uploaded
        if 'images[]' not in request.files:
            flash('No images selected. Please upload at least one image.', 'error')
            return redirect(request.url)
        
        images = request.files.getlist('images[]')
        results = []
        
        # FORCE ENGLISH TRANSCRIPTION
        target_language = "auto"
        translate_to_english = True
        
        print(f"Language settings - Target: {target_language}, Translate to English: {translate_to_english}")
        
        # Filter out empty files
        valid_images = [img for img in images if img and img.filename != '']
        
        if not valid_images:
            flash('No valid images selected. Please check your files.', 'error')
            return redirect(request.url)
        
        print(f"Processing {len(valid_images)} image(s)...")
        
        for i, image in enumerate(valid_images):
            if allowed_image_file(image.filename):
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                image_ext = image.filename.rsplit('.', 1)[1].lower()
                image_filename = f"image_{unique_id}.{image_ext}"
                
                # Save image
                image_path = save_uploaded_file(
                    image, 
                    app.config['UPLOAD_FOLDER'], 
                    image_filename
                )
                
                # Process recorded audio (base64 data)
                audio_transcription = None
                recorded_audio_key = f'recorded_audio_{i+1}'
                
                if recorded_audio_key in request.form and request.form[recorded_audio_key].strip():
                    try:
                        audio_data = request.form[recorded_audio_key]
                        
                        if audio_data.strip():
                            # Save as WebM (browser default)
                            audio_filename = f"audio_recorded_{unique_id}.webm"
                            
                            print(f"Saving recorded audio for image {i+1}...")
                            
                            # Save the audio file
                            audio_path = save_base64_audio(
                                audio_data,
                                app.config['UPLOAD_FOLDER'],
                                audio_filename
                            )
                            
                            print(f"✓ Audio saved: {audio_path}")
                            
                            # Convert to WAV using FFmpeg
                            wav_filename = f"audio_recorded_{unique_id}.wav"
                            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
                            
                            print("Converting audio to WAV format...")
                            final_audio_path = convert_audio_to_wav(audio_path, wav_path)
                            
                            # Use converted WAV if available
                            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                                final_audio_path = wav_path
                                print(f"✓ Using converted WAV file: {final_audio_path}")
                                # Clean up original WebM file
                                try:
                                    if os.path.exists(audio_path):
                                        os.remove(audio_path)
                                        print(f"✓ Cleaned up original WebM file: {audio_path}")
                                except Exception as e:
                                    print(f"Note: Could not clean up WebM file: {e}")
                            else:
                                final_audio_path = audio_path
                                print(f"✓ Using original audio file: {final_audio_path}")
                            
                            print(f"Processing recorded audio for image {i+1}...")
                            
                            # Transcribe audio if model is available
                            if processor and model:
                                try:
                                    start_time = time.time()
                                    audio_transcription = transcribe_audio_simple(
                                        final_audio_path, 
                                        processor, 
                                        model,
                                        target_language=target_language,
                                        translate_to_english=translate_to_english
                                    )
                                    end_time = time.time()
                                    print(f"✓ Recorded audio transcription completed in {end_time - start_time:.2f} seconds")
                                    print(f"✓ Final transcription: {audio_transcription}")
                                    
                                    # Clean up audio file after processing
                                    try:
                                        if os.path.exists(final_audio_path):
                                            os.remove(final_audio_path)
                                            print(f"✓ Cleaned up audio file: {final_audio_path}")
                                    except Exception as e:
                                        print(f"Note: Could not clean up audio file: {e}")
                                        
                                except Exception as e:
                                    audio_transcription = f"Transcription error: {str(e)}"
                                    print(f"✗ Recorded audio transcription failed: {e}")
                                    print(traceback.format_exc())
                            else:
                                audio_transcription = "Transcription model not available"
                                print("✗ Audio transcription not available - model not loaded")
                        else:
                            print(f"No recorded audio data for image {i+1}")
                    
                    except Exception as e:
                        error_msg = f"Error processing recorded audio: {str(e)}"
                        print(error_msg)
                        print(traceback.format_exc())
                        audio_transcription = error_msg
                
                # Prepare result
                result = {
                    'image_url': f"/static/uploads/{image_filename}",
                    'image_name': image.filename,
                    'transcription': audio_transcription,
                    'has_audio': audio_transcription is not None
                }
                results.append(result)
                print(f"✓ Processed image {i+1}: {image.filename}")
        
        if not results:
            flash('No valid images were processed. Please check your files.', 'error')
            return redirect(request.url)
            
        flash(f'Successfully processed {len(results)} image(s)!', 'success')
        
        # Return to template
        languages = [
            ('auto', 'Auto-detect Language'),
            ('english', 'English'),
            ('spanish', 'Spanish'),
            ('french', 'French'),
            ('german', 'German'),
            ('chinese', 'Chinese'),
            ('japanese', 'Japanese'),
            ('korean', 'Korean'),
            ('hindi', 'Hindi'),
            ('arabic', 'Arabic'),
            ('russian', 'Russian'),
            ('portuguese', 'Portuguese'),
            ('italian', 'Italian')
        ]
        
        return render_template('index.html', 
                             results=results, 
                             languages=languages,
                             selected_language=target_language,
                             translate_to_english=translate_to_english)
    
    except Exception as e:
        error_msg = f'Error processing files: {str(e)}'
        print(f"✗ {error_msg}")
        print(traceback.format_exc())
        flash(error_msg, 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        import torch
        pytorch_available = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pytorch_available = False
        device = "cpu"
    
    return jsonify({
        'status': 'healthy',
        'pytorch_available': pytorch_available,
        'huggingface_token_set': bool(HUGGINGFACE_TOKEN),
        'device': device,
        'model_loaded': processor is not None and model is not None,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'ffmpeg_available': check_ffmpeg_available()
    })

if __name__ == '__main__':
    print("🚀 Starting Voice-to-Image Metadata Application...")
    
    # Check FFmpeg availability
    if check_ffmpeg_available():
        print("✓ FFmpeg is available")
    else:
        print("⚠ FFmpeg is not available - audio conversion may not work properly")
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        print(f"✓ Created upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Initialize model
    initialize_model()
    
    print("✓ Flask application starting...")
    print("📱 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
