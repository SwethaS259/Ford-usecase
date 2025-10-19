# utils.py
import os
import uuid
import tempfile
import base64
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')

# Try to import torch, but provide fallback if not available
try:
    import torch
    import numpy as np
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from huggingface_hub import login
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch/Transformers not available: {e}")
    TORCH_AVAILABLE = False

# Check for GPU availability if torch is available
if TORCH_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
else:
    device = "cpu"
    print("PyTorch not available, running in fallback mode")

# Load Whisper model with authentication
def load_whisper_model():
    """Load Whisper model and processor with Hugging Face token"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot load Whisper model")
        return None, None
        
    try:
        # Login to Hugging Face if token is provided
        if HUGGINGFACE_TOKEN:
            print("Authenticating with Hugging Face...")
            login(token=HUGGINGFACE_TOKEN)
        else:
            print("No Hugging Face token provided. Using anonymous access (may have rate limits).")
        
        print("Loading Whisper medium model... This may take a few minutes.")
        
        # Use Whisper medium model as requested
        model_name = "openai/whisper-medium"
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to appropriate device
        model = model.to(device)
        
        print(f"✓ {model_name} loaded successfully!")
        return processor, model
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

def convert_audio_to_wav(input_path, output_path):
    """Convert any audio format to WAV using FFmpeg"""
    try:
        print(f"Converting audio to WAV: {input_path} -> {output_path}")
        
        # Use FFmpeg to convert to WAV format
        result = subprocess.run([
            'ffmpeg',
            '-i', input_path,      # Input file
            '-acodec', 'pcm_s16le', # Audio codec: 16-bit PCM
            '-ac', '1',            # Mono audio
            '-ar', '16000',        # Sample rate: 16kHz
            '-y',                  # Overwrite output file
            output_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Successfully converted to WAV: {output_path}")
                return output_path
            else:
                raise Exception("Output file was not created properly")
        else:
            raise Exception(f"FFmpeg error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg conversion timed out")
        return input_path
    except Exception as e:
        print(f"✗ Audio conversion failed: {e}")
        return input_path

def transcribe_audio(audio_path, processor, model, target_language="english", translate_to_english=False):
    """Unified audio transcription - ALWAYS translates to English"""
    if not TORCH_AVAILABLE:
        return "PyTorch not available - cannot transcribe audio"
        
    try:
        print(f"=== STARTING AUDIO TRANSCRIPTION ===")
        print(f"Audio path: {audio_path}")
        
        # Check if file exists and has content
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            print(f"✗ {error_msg}")
            return error_msg
            
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = "Audio file is empty"
            print(f"✗ {error_msg}")
            return error_msg
        
        # Convert to WAV if not already WAV
        final_audio_path = audio_path
        if not audio_path.lower().endswith('.wav'):
            print("Converting audio to WAV format for compatibility...")
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            final_audio_path = convert_audio_to_wav(audio_path, wav_path)
        
        # Import librosa here to avoid issues if not available
        try:
            import librosa
            print("✓ Librosa imported successfully")
        except ImportError as e:
            error_msg = f"Librosa not available for audio processing: {e}"
            print(f"✗ {error_msg}")
            return error_msg
        
        # Process audio file
        try:
            print("Loading audio with librosa...")
            
            # Load audio with librosa
            audio_array, sampling_rate = librosa.load(final_audio_path, sr=16000, mono=True)
            
            if len(audio_array) == 0:
                error_msg = "Audio file contains no data after loading"
                print(f"✗ {error_msg}")
                return error_msg
                
            print(f"✓ Audio loaded successfully: {len(audio_array)} samples, {sampling_rate} Hz")
            
            # Check audio duration
            duration = len(audio_array) / sampling_rate
            print(f"Audio duration: {duration:.2f} seconds")
            
            if duration < 0.1:
                error_msg = f"Audio too short (less than 0.1 seconds): {duration:.2f}s"
                print(f"✗ {error_msg}")
                return error_msg
            
            # Check if audio is not silent
            max_amplitude = np.max(np.abs(audio_array))
            print(f"Max audio amplitude: {max_amplitude}")
            
            if max_amplitude == 0:
                error_msg = "Audio appears to be silent (max amplitude is 0)"
                print(f"✗ {error_msg}")
                return error_msg
            
            # Normalize audio if not silent
            if max_amplitude > 0:
                print("Normalizing audio...")
                audio_array = audio_array / max_amplitude
            
            print("Processing audio with Whisper processor...")
            # Process the audio with Whisper
            inputs = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            
            print(f"Input features shape: {inputs.input_features.shape}")
            
            # Move to appropriate device
            input_features = inputs.input_features.to(device)
            print(f"✓ Moved inputs to device: {device}")
            
            # FORCE ENGLISH TRANSCRIPTION - Always translate to English
            print("Setting up forced decoder for English translation...")
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="translate")
            
            print("Generating transcription...")
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=5,
                    temperature=0.0,
                    patience=2.0
                )
            
            print("✓ Transcription generated successfully")
            
            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            result = transcription[0].strip() if transcription else "No transcription available"
            print(f"✓ FINAL TRANSCRIPTION RESULT: '{result}'")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(f"✗ {error_msg}")
            print("Full traceback:")
            import traceback
            traceback.print_exc()
            return f"Audio processing error: {str(e)}"
            
    except Exception as e:
        error_msg = f"Error in transcription: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

# Make transcribe_audio_simple an alias of transcribe_audio
def transcribe_audio_simple(audio_path, processor, model, target_language="english", translate_to_english=False):
    """Alias for transcribe_audio - both functions now work identically"""
    return transcribe_audio(audio_path, processor, model, target_language, translate_to_english)

# Validate image file
def allowed_image_file(filename):
    """Check if the file is an allowed image type"""
    if not filename or filename == '':
        return False
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Save uploaded file
def save_uploaded_file(file, upload_folder, filename):
    """Save uploaded file to specified folder"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path

# Save base64 audio data
def save_base64_audio(audio_data, upload_folder, filename):
    """Save base64 audio data to file with proper validation"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, filename)
    
    try:
        # Remove data URL prefix if present
        if audio_data.startswith('data:audio/'):
            audio_data = audio_data.split(',')[1]
        elif audio_data.startswith('data:audio/webm;base64,'):
            audio_data = audio_data.split(',')[1]
        elif audio_data.startswith('data:audio/wav;base64,'):
            audio_data = audio_data.split(',')[1]
        elif audio_data.startswith('data:'):
            # Generic data URL handling
            audio_data = audio_data.split(',')[1]
        
        # Validate base64 data
        if not audio_data.strip():
            raise ValueError("Empty audio data")
            
        # Decode and save
        audio_bytes = base64.b64decode(audio_data)
        
        # Check if decoded data has reasonable size
        if len(audio_bytes) < 100:
            raise ValueError("Audio data too small to be valid")
            
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✓ Saved base64 audio: {file_path} ({len(audio_bytes)} bytes)")
        return file_path
        
    except Exception as e:
        print(f"✗ Error saving base64 audio: {e}")
        # Clean up invalid file if it was created
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

def check_ffmpeg_available():
    """Check if FFmpeg is available in the system"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False
