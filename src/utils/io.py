import cv2
import numpy as np

def load_video_frames(video_path, num_frames=16):
    """
    Load and sample frames from a video.
    
    Args:
        video_path (str): Path to video file.
        num_frames (int): Number of frames to sample.
    
    Returns:
        np.ndarray: Array of shape [num_frames, height, width, 3]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Sample frames uniformly
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    elif len(frames) > num_frames:
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return np.array(frames)