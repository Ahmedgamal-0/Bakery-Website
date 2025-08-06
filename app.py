from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import mediapipe as mp
import logging
import math
from collections import deque
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize models
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Try to load the best model, fallback to YOLOv8 if not available
try:
    yolo_model = YOLO("best1.pt")
    logging.info("Loaded custom golf ball model: best1.pt")
except:
    try:
        yolo_model = YOLO("yolov8s.pt")
        logging.info("Loaded YOLOv8s model as fallback")
    except:
        yolo_model = None
        logging.error("No YOLO model available")

# Constants
MIN_BALL_AREA = 30
MAX_BALL_AREA = 5000
ASPECT_RATIO_THRESHOLD = 1.2
MIN_TRAJECTORY_POINTS = 3
MAX_TRAJECTORY_POINTS = 100
MIN_SWING_DISTANCE = 10

# Physics constants
GRAVITY = 9.8  # m/s²
PIXELS_PER_METER = 100  # Calibration factor

def create_kalman_filter():
    """Create a Kalman filter for ball tracking"""
    kf = cv2.KalmanFilter(4, 2)  # 4 states: x, y, vx, vy
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  # Reduced noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # Reduced noise
    kf.errorCovPre = np.eye(4, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    return kf

def needs_rotation(width, height):
    """Determine if video needs rotation"""
    return (width / height) > ASPECT_RATIO_THRESHOLD

def detect_ball_yolo(red_zone, rx, ry, frame_count):
    """Detect ball using YOLO model"""
    if red_zone is None or red_zone.size == 0:
        return None, None
    
    try:
        # Convert to RGB for YOLO
        rgb_zone = cv2.cvtColor(red_zone, cv2.COLOR_BGR2RGB)
        detections = yolo_model.predict(source=rgb_zone, conf=0.2, verbose=False)[0]
        
        best_center = None
        best_box = None
        max_conf = 0
        
        for box in detections.boxes:
            # Check if it's a ball (class 0 for custom model, 32 for COCO sports ball)
            class_id = int(box.cls[0])
            if class_id in [0, 32]:  # Golf ball or sports ball
                conf = float(box.conf[0])
                if conf > max_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    if MIN_BALL_AREA < area < MAX_BALL_AREA:
                        max_conf = conf
                        cx = (x1 + x2) // 2 + rx
                        cy = (y1 + y2) // 2 + ry
                        best_center = (cx, cy)
                        best_box = [x1 + rx, y1 + ry, x2 + rx, y2 + ry]
        
        if best_center:
            logging.info(f"Frame {frame_count}: Ball detected at {best_center} (conf: {max_conf:.2f})")
            return best_center, best_box
        
    except Exception as e:
        logging.error(f"Frame {frame_count}: YOLO detection error: {e}")
    
    return None, None

def calculate_trajectory_physics(trajectory_points, fps=30):
    """Calculate initial velocity and angle from trajectory points with improved noise handling"""
    if len(trajectory_points) < 3:
        return None, None, None
    
    # Convert to numpy arrays
    points = np.array(trajectory_points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Use only the first few points for initial velocity (more stable)
    num_points = min(5, len(trajectory_points))
    
    # Simple velocity calculation using first few points
    if num_points >= 3:
        # Use first 3 points for most stable calculation
        dx = x[2] - x[0]  # Change over 2 frames
        dy = y[2] - y[0]
        dt = 2  # 2 frame difference
        
        avg_dx = dx / dt
        avg_dy = dy / dt
    else:
        # Fallback to simple difference
        avg_dx = np.mean(np.diff(x[:num_points]))
        avg_dy = np.mean(np.diff(y[:num_points]))
    
    # Apply velocity limits to prevent unrealistic values
    MAX_VELOCITY_PX_FRAME = 15  # Reduced to 15 pixels per frame
    avg_dx = np.clip(avg_dx, -MAX_VELOCITY_PX_FRAME, MAX_VELOCITY_PX_FRAME)
    avg_dy = np.clip(avg_dy, -MAX_VELOCITY_PX_FRAME, MAX_VELOCITY_PX_FRAME)
    
    # Convert to pixels per second
    vx0 = avg_dx * fps
    vy0 = avg_dy * fps
    
    # Calculate initial velocity magnitude and angle
    v0 = np.sqrt(vx0**2 + vy0**2)
    angle = np.arctan2(vy0, vx0) * 180 / np.pi  # Convert to degrees
    
    # Apply more conservative velocity limits
    MAX_REALISTIC_VELOCITY = 450  # Reduced to 450 px/s (4.5 m/s reasonable for video analysis)
    if v0 > MAX_REALISTIC_VELOCITY:
        scale_factor = MAX_REALISTIC_VELOCITY / v0
        v0 = MAX_REALISTIC_VELOCITY
        vx0 *= scale_factor
        vy0 *= scale_factor
    
    logging.info(f"Trajectory physics: v0={v0:.1f} px/s, angle={angle:.1f}°, vx0={vx0:.1f}, vy0={vy0:.1f}")
    
    return v0, angle, (vx0, vy0)

def predict_trajectory(start_point, velocity, gravity_effect, frame_height, num_points=50):
    """Predict future trajectory based on initial conditions until ball hits ground"""
    trajectory = []
    x, y = start_point
    vx, vy = velocity
    
    for i in range(num_points):
        # Update position
        x += vx
        y += vy
        
        # Apply gravity effect (positive because y increases downward in image coordinates)
        vy += abs(gravity_effect)  # Make sure gravity pulls downward
        
        trajectory.append((x, y))
        
        # Stop if ball goes below frame or too far horizontally
        if y > frame_height or x < 0 or x > frame_height * 2:  # Assume reasonable bounds
            break
    
    return trajectory

def draw_physics_based_trajectory(frame, trajectory_points, initial_position, fps):
    """Draw trajectory with physics-based prediction"""
    if not trajectory_points and not initial_position:
        return frame
    
    frame_height, frame_width = frame.shape[:2]
    
    # Always include initial position at the start
    all_points = []
    if initial_position:
        all_points.append((int(initial_position[0]), int(initial_position[1])))
    
    # Add trajectory points
    for point in trajectory_points:
        all_points.append((int(point[0]), int(point[1])))
    
    if len(all_points) < 2:
        return frame
    
    # Convert points to numpy array
    points = np.array(all_points, dtype=np.int32)
    
    # Draw much thicker trajectory line for observed points
    cv2.polylines(frame, [points], False, (0, 255, 0), 8)
    
    # Calculate physics if we have enough points
    if len(trajectory_points) >= 3:
        try:
            physics_result = calculate_trajectory_physics(trajectory_points, fps)
            if physics_result is not None:
                v0, angle, (vx0, vy0) = physics_result
                
                # Draw physics info on frame
                info_text = f"v0: {v0:.0f} px/s"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                
                angle_text = f"Angle: {angle:.1f}°"
                cv2.putText(frame, angle_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                
                # Predict future trajectory with gravity
                if len(trajectory_points) >= 3:
                    last_point = trajectory_points[-1]
                    # Increase gravity effect for more realistic parabola
                    gravity_effect = (GRAVITY * PIXELS_PER_METER * 2) / (fps**2)  # Doubled gravity effect
                    predicted_points = predict_trajectory(
                        last_point,
                        (vx0/fps, vy0/fps),  # Convert to px/frame
                        gravity_effect,
                        frame_height
                    )
                    
                    # Draw predicted trajectory in thick red line
                    if len(predicted_points) > 1:
                        future_points = [last_point]
                        for pt in predicted_points:
                            future_points.append((int(pt[0]), int(pt[1])))
                        
                        future_array = np.array(future_points, dtype=np.int32)
                        cv2.polylines(frame, [future_array], False, (0, 0, 255), 8)  # Much thicker red line
                        
                        # Find and mark apex (highest point) - only if trajectory goes up first
                        if len(future_points) > 5:
                            y_values = [pt[1] for pt in future_points]
                            min_y_idx = np.argmin(y_values)  # Lowest Y = highest point in image
                            
                            # Only mark apex if it's not at the start (ball actually went up)
                            if min_y_idx > 2:
                                apex_x, apex_y = future_points[min_y_idx]
                                cv2.circle(frame, (int(apex_x), int(apex_y)), 20, (255, 0, 0), -1)
                                cv2.circle(frame, (int(apex_x), int(apex_y)), 15, (255, 255, 255), -1)
                                cv2.putText(frame, 'APEX', (int(apex_x) + 25, int(apex_y) - 25), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                        
                        # Mark landing point (last point of trajectory)
                        if len(future_points) > 10:
                            landing_x, landing_y = future_points[-1]
                            cv2.circle(frame, (int(landing_x), int(landing_y)), 20, (0, 255, 255), -1)
                            cv2.circle(frame, (int(landing_x), int(landing_y)), 15, (0, 0, 0), -1)
                            cv2.putText(frame, 'LAND', (int(landing_x) + 25, int(landing_y) + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                        
        except Exception as e:
            logging.error(f"Physics calculation error: {e}")
    
    return frame

@app.route('/predict-video', methods=['POST'])
def predict_video():
    """Main video processing endpoint"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_input.name)
    logging.info(f"Processing video: {temp_input.name}")

    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video file'}), 400

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    should_rotate = needs_rotation(width, height)
    if should_rotate:
        width, height = height, width
        logging.info("Video will be rotated 90° clockwise")

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize tracking
    kalman = create_kalman_filter()
    trajectory_points = deque(maxlen=MAX_TRAJECTORY_POINTS)
    tracking_started = False
    ball_motion_started = False
    initial_position = None
    frame_count = 0
    velocity_history = deque(maxlen=5)  # Store recent velocities for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Process pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose_estimator.process(frame_rgb)

        best_center = None
        best_box = None

        if pose_res.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get person bounding box
            landmarks = pose_res.pose_landmarks.landmark
            x_coords = [int(lm.x * width) for lm in landmarks]
            y_coords = [int(lm.y * height) for lm in landmarks]
            x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))
            
            # Draw person bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, 'Player', (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Define detection zone (to the right of player)
            margin = 80
            red_width = (x_max - x_min) + 2 * margin
            red_x1 = min(width - 1, x_max + 30)
            red_x2 = min(width, red_x1 + red_width)
            red_y1 = max(0, y_min - 60)
            red_y2 = min(height, y_max + 60)
            
            if red_x2 > red_x1 and red_y2 > red_y1:
                red_zone = frame[red_y1:red_y2, red_x1:red_x2]
                cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Detection Zone', (red_x1, red_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Ball detection using YOLO only
                if yolo_model:
                    center, box = detect_ball_yolo(red_zone, red_x1, red_y1, frame_count)
                    if center:
                        best_center = center
                        best_box = box

        # Kalman filter update and tracking with improved velocity handling
        if best_center:
            current_pos = np.array([[np.float32(best_center[0])], [np.float32(best_center[1])]], dtype=np.float32)
            
            if not tracking_started:
                # Extract numeric coordinates
                bx = float(best_center[0]) if np.isscalar(best_center[0]) else float(best_center[0][0])
                by = float(best_center[1]) if np.isscalar(best_center[1]) else float(best_center[1][0])

                # Initialize Kalman state with zero velocity
                kalman.statePre = np.array([[bx], [by], [0.0], [0.0]], dtype=np.float32)
                kalman.statePost = np.array([[bx], [by], [0.0], [0.0]], dtype=np.float32)

                tracking_started = True
                initial_position = (bx, by)
                filtered_x, filtered_y = bx, by
                trajectory_points.append((filtered_x, filtered_y))
                logging.info(f"Frame {frame_count}: Tracking started at ({filtered_x:.1f}, {filtered_y:.1f})")
            else:
                # Build measurement
                meas_x = float(current_pos[0]) if np.isscalar(current_pos[0]) else float(current_pos[0][0])
                meas_y = float(current_pos[1]) if np.isscalar(current_pos[1]) else float(current_pos[1][0])
                measurement = np.array([[meas_x], [meas_y]], dtype=np.float32)

                                 # Predict then correct with velocity limiting
                 try:
                     pred = kalman.predict()
                     
                     # Limit predicted velocities to prevent runaway values
                     MAX_VEL = 20.0  # Reduced to 20 pixels per frame
                     if abs(pred[2, 0]) > MAX_VEL:
                         pred[2, 0] = np.sign(pred[2, 0]) * MAX_VEL
                     if abs(pred[3, 0]) > MAX_VEL:
                         pred[3, 0] = np.sign(pred[3, 0]) * MAX_VEL
                     
                     kalman.statePost = pred.copy()  # Use limited prediction
                     kalman.correct(measurement)
                     
                     # Further limit corrected velocities
                     if abs(kalman.statePost[2, 0]) > MAX_VEL:
                         kalman.statePost[2, 0] = np.sign(kalman.statePost[2, 0]) * MAX_VEL
                     if abs(kalman.statePost[3, 0]) > MAX_VEL:
                         kalman.statePost[3, 0] = np.sign(kalman.statePost[3, 0]) * MAX_VEL
                    
                    filtered_x = float(kalman.statePost[0, 0])
                    filtered_y = float(kalman.statePost[1, 0])
                    
                except Exception as e:
                    logging.warning(f"Kalman predict/correct failed at frame {frame_count}: {e}")
                    filtered_x = meas_x
                    filtered_y = meas_y

                # Detect motion start using initial_position
                if not ball_motion_started and initial_position is not None:
                    distance = math.hypot(filtered_x - initial_position[0], filtered_y - initial_position[1])
                    if distance > MIN_SWING_DISTANCE:
                        ball_motion_started = True
                        logging.info(f"Frame {frame_count}: Ball motion detected (dist={distance:.1f})")

                trajectory_points.append((filtered_x, filtered_y))

            # Draw ball detection
            if best_box:
                cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), 
                             (0, 255, 255), 2)
                cv2.putText(frame, 'Ball', (best_box[0], best_box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif tracking_started:
            # Predict ball position when not detected
            pred_state = kalman.predict()
            pred_center = (int(pred_state[0, 0]), int(pred_state[1, 0]))
            
            # Add predicted position to trajectory
            trajectory_points.append((pred_center[0], pred_center[1]))
            
            # Draw predicted position
            cv2.circle(frame, pred_center, 5, (0, 255, 0), -1)
            cv2.putText(frame, 'Predicted', (pred_center[0] + 10, pred_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw trajectory with physics-based prediction
        frame = draw_physics_based_trajectory(frame, list(trajectory_points), initial_position, fps)

        out.write(frame)

    cap.release()
    out.release()

    # Clean up temporary files
    try:
        os.remove(temp_input.name)
    except:
        pass

    logging.info(f"Processing complete. Output saved to {temp_output.name}")
    return send_file(temp_output.name, mimetype='video/mp4', 
                     as_attachment=True, download_name='golf_swing_analysis.mp4')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Golf ball tracking API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)