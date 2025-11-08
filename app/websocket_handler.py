import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from flask_socketio import emit
from app import socketio, emotion_service
import logging
from flask import request

logger = logging.getLogger('websocket_handler')

# ===============================================
# SOCKETIO EVENTS
# ===============================================

@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    logger.info("Client connected: %s", request.sid)
    if not emotion_service.is_monitoring:
        emotion_service.start_monitoring()
        emit('server_message', {'data': 'Monitoring session started.'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected: %s", request.sid)
    # Optionally stop monitoring if no clients are left

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Receive video frame (base64 image) for face emotion analysis.
    
    data = { 'image': <base64 encoded jpeg> }
    """
    if not emotion_service.is_monitoring:
        return

    try:
        # Decode base64 to image (PIL)
        b64_img = data['image'].split(',')[1] # Remove data URL prefix
        image_bytes = base64.b64decode(b64_img)
        img = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV numpy array (BGR)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Process frame in the monitoring service
        face_result = emotion_service.process_face_frame(frame)
        
        # Only proceed with fusion if face detection occurred and voice data is expected
        if face_result:
            # We assume a voice result is also available (or is None)
            # NOTE: In a real system, voice and face would run asynchronously. 
            # Here we fuse the latest known voice result (which the service tracks)
            
            # For simplicity in this demo, we use the service's internal fusion logic
            # which happens within update_emotion_state. 
            # A more complex system would queue predictions and fuse independently.
            
            # Get latest voice prediction from smoother (if it was updated recently)
            # This is handled internally by the service when process_voice_sample is called.
            
            # Update and get the intervention decision
            intervention_result = emotion_service.update_emotion_state(face_result)
            
            if intervention_result:
                # Emit the fused, smooth, and action-ready result back to the frontend
                emit('emotion_update', intervention_result)
                
                # Check for intervention level change (e.g., switch game)
                if intervention_result['level_changed']:
                    emit('intervention_trigger', {
                        'level': intervention_result['level'],
                        'action': intervention_result['action'],
                        'duration': intervention_result['duration']
                    })

    except Exception as e:
        logger.error("Error processing video frame: %s", e)
        emit('server_message', {'data': f'Error: {e}'})


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """
    Receive audio chunk (numpy array or list) for voice emotion analysis.
    
    data = { 'audio': <list of floats/ints> }
    """
    if not emotion_service.is_monitoring:
        return

    try:
        # Convert list to numpy array (should be float or int PCM data)
        audio_buffer = np.array(data['audio'], dtype=np.float32) 
        
        # Process audio sample
        voice_result = emotion_service.process_voice_sample(
            audio_buffer,
            sample_rate=emotion_service.config.VOICE_SAMPLE_RATE # Use config SR
        )
        
        if voice_result:
            # Update the service state. The service handles fusion logic.
            intervention_result = emotion_service.update_emotion_state(voice_result)
            
            if intervention_result:
                # Emit the fused, smooth, and action-ready result back to the frontend
                emit('emotion_update', intervention_result)
                
                # Check for intervention level change
                if intervention_result['level_changed']:
                    emit('intervention_trigger', {
                        'level': intervention_result['level'],
                        'action': intervention_result['action'],
                        'duration': intervention_result['duration']
                    })

    except Exception as e:
        logger.error("Error processing audio chunk: %s", e)
        emit('server_message', {'data': f'Error: {e}'})