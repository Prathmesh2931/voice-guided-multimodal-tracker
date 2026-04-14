#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class EnhancedYOLOTracker(Node):
    def __init__(self):
        super().__init__('enhanced_yolo_tracker')

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)

        # Voice command subscribers
        self.target_object_sub = self.create_subscription(String, '/target_object', self.target_object_callback, 10)
        self.control_mode_sub = self.create_subscription(String, '/control_mode', self.control_mode_callback, 10)
        self.search_mode_sub = self.create_subscription(Bool, '/search_mode', self.search_mode_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_pub = self.create_publisher(Bool, '/object_detected', 10)
        self.distance_pub = self.create_publisher(Float32, '/object_distance', 10)
        self.state_pub = self.create_publisher(String, '/robot_state', 10)
        self.detection_image_pub = self.create_publisher(Image, '/detection_image', qos_profile)

        # State variables
        self.rgb_image = None
        self.depth_image = None
        self.laser_data = None

        # Tracking variables
        self.target_object = "person"
        self.tracking_mode = "yolo"       # "yolo" or "color"
        self.control_mode = "searching"   # "searching", "following", "stopped", "paused"
        self.search_mode = True

        # Detection variables
        self.object_detected = False
        self.object_distance = float('inf')
        self.object_position = [0, 0]
        self.detection_confidence = 0.0
        self.min_confidence = 0.1
        self.target_distance = 0.2
        self.dist_target_tolerance = 0.05

        # Movement parameters
        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.4
        self.search_angular_speed = 0.3

        # Control gains
        self.angular_gain = 1.0
        self.linear_gain = 0.5
        self.alignment_threshold = 0.25

        # Safety
        self.min_obstacle_distance = 0.1
        self.emergency_stop = False

        # State tracking
        self.current_state = "IDLE"
        self.last_state = ""
        self.state_change_time = time.time()
        self.no_depth_count = 0

        # Color tracking parameters — widened HSV ranges for real-world lighting
        self.color_ranges = {
            "red_object": {
                "lower1": np.array([0, 80, 80]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 80, 80]),
                "upper2": np.array([180, 255, 255])
            },
            "blue_object": {
                # FIX: widened from [100,100,100]→[130,255,255]
                "lower1": np.array([90, 50, 50]),
                "upper1": np.array([140, 255, 255]),
                "lower2": None,
                "upper2": None
            },
            "green_object": {
                # FIX: widened from [40,100,100]
                "lower1": np.array([35, 50, 50]),
                "upper1": np.array([85, 255, 255]),
                "lower2": None,
                "upper2": None
            }
        }

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # YOLO class names
        self.class_names = self.yolo_model.names

        self.get_logger().info('🤖 Enhanced YOLO Tracker initialized')
        self.get_logger().info(f'📏 Target distance: {self.target_distance}m (±{self.dist_target_tolerance}m)')
        self.get_logger().info(f'🎯 Alignment threshold: {self.alignment_threshold}')

    # ─────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────

    def update_state(self, new_state):
        """Update and publish robot state."""
        if new_state != self.current_state:
            self.last_state = self.current_state
            self.current_state = new_state
            self.state_change_time = time.time()

            state_msg = String()
            state_msg.data = new_state
            self.state_pub.publish(state_msg)

            state_emojis = {
                "SEARCHING":       "🔍",
                "OBJECT_DETECTED": "👁️",
                "ALIGNING":        "🎯",
                "APPROACHING":     "➡️",
                "AT_TARGET":       "✅",
                "REVERSING":       "⬅️",
                "STOPPED":         "🛑",
                "PAUSED":          "⏸️",
                "EMERGENCY_STOP":  "⚠️",
                "NO_DEPTH_DATA":   "❓"
            }
            emoji = state_emojis.get(new_state, "🤖")
            self.get_logger().info(f'{emoji} STATE: {new_state}')

            if new_state == "AT_TARGET":
                self.get_logger().info(f'✅ Reached target distance: {self.object_distance:.2f}m')
            elif new_state == "APPROACHING":
                self.get_logger().info(f'➡️  Moving forward - Distance: {self.object_distance:.2f}m')
            elif new_state == "REVERSING":
                self.get_logger().info(f'⬅️  Too close! Backing up - Distance: {self.object_distance:.2f}m')
            elif new_state == "ALIGNING":
                image_center_x = self.rgb_image.shape[1] // 2 if self.rgb_image is not None else 0
                error = (
                    (self.object_position[0] - image_center_x) / (self.rgb_image.shape[1] / 2)
                    if self.rgb_image is not None else 0
                )
                direction = "RIGHT" if error > 0 else "LEFT"
                self.get_logger().info(f'🎯 Aligning to {direction} - Error: {abs(error):.2f}')

    # ─────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────

    def target_object_callback(self, msg):
        """
        Update target object from voice command.

        FIX 1: Set control_mode = "following" so the control loop enters
                follow_behavior instead of spinning forever in searching.
        FIX 2: Reset stale detection state so the robot re-acquires the
                new target cleanly.
        """
        new_target = msg.data.lower()

        # --- Reset state for new target (FIX 2) ---
        self.object_detected = False
        self.object_distance = float('inf')
        self.control_mode = "searching"

        if new_target in self.color_ranges:
            self.target_object = new_target
            self.tracking_mode = "color"
            self.control_mode = "following"   # FIX 1
            self.get_logger().info(f'🎨 Switching to color tracking: {new_target}')
        else:
            matched = False
            for class_id, class_name in self.class_names.items():
                if new_target in class_name.lower() or class_name.lower() in new_target:
                    self.target_object = class_name
                    self.tracking_mode = "yolo"
                    self.control_mode = "following"   # FIX 1
                    self.get_logger().info(f'🤖 Switching to YOLO tracking: {class_name}')
                    matched = True
                    break

            if not matched:
                self.control_mode = "searching"
                self.get_logger().warn(f'❓ Unknown object: {new_target}, staying in search mode')

    def control_mode_callback(self, msg):
        """
        Update control mode from external topic.
        FIX 1 (global): remap legacy 'color_tracking' → 'following' so the
        control loop never receives an unhandled mode string.
        """
        incoming = msg.data.lower()
        if incoming == "color_tracking":
            self.control_mode = "following"
            self.get_logger().info('🎮 Control mode: color_tracking → remapped to following')
        else:
            self.control_mode = incoming
            self.get_logger().info(f'🎮 Control mode: {self.control_mode}')

    def search_mode_callback(self, msg):
        """Update search mode."""
        self.search_mode = msg.data
        if self.search_mode:
            self.control_mode = "searching"

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.detect_objects()
        except Exception as e:
            self.get_logger().error(f'❌ RGB callback error: {e}')

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f'❌ Depth callback error: {e}')

    def scan_callback(self, msg):
        self.laser_data = msg
        self.check_obstacles()

    # ─────────────────────────────────────────────
    # Detection
    # ─────────────────────────────────────────────

    def detect_objects(self):
        """Detect objects using YOLO or color detection."""
        if self.rgb_image is None:
            return
        if self.tracking_mode == "yolo":
            self.detect_yolo_objects()
        elif self.tracking_mode == "color":
            self.detect_color_objects()
        self.publish_detection_image()

    def detect_yolo_objects(self):
        """YOLO-based object detection."""
        try:
            results = self.yolo_model(self.rgb_image, verbose=False)

            self.object_detected = False
            best_detection = None
            best_confidence = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names[class_id]

                        if (class_name.lower() == self.target_object.lower() and
                                confidence > self.min_confidence and
                                confidence > best_confidence):

                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            best_detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }
                            best_confidence = confidence

            if best_detection:
                self.object_detected = True
                self.object_position = best_detection['center']
                self.detection_confidence = best_confidence
                self.calculate_object_distance()
                self.draw_yolo_detection(best_detection)

        except Exception as e:
            self.get_logger().error(f'❌ YOLO detection error: {e}')

    def detect_color_objects(self):
        """Color-based object detection."""
        try:
            if self.target_object not in self.color_ranges:
                return

            hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
            color_range = self.color_ranges[self.target_object]

            mask1 = cv2.inRange(hsv, color_range["lower1"], color_range["upper1"])
            if color_range["lower2"] is not None:
                mask2 = cv2.inRange(hsv, color_range["lower2"], color_range["upper2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = mask1

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Debug: log contour count to help diagnose invisible-object issues
            self.get_logger().debug(f'Contours found: {len(contours)}')

            self.object_detected = False
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 1000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    self.object_detected = True
                    self.object_position = [center_x, center_y]
                    self.detection_confidence = min(area / 10000.0, 1.0)
                    self.calculate_object_distance()

                    cv2.rectangle(self.rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        self.rgb_image,
                        f'{self.target_object}: {self.detection_confidence:.2f}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

        except Exception as e:
            self.get_logger().error(f'❌ Color detection error: {e}')

    def calculate_object_distance(self):
        """Calculate distance to detected object."""
        if self.depth_image is None or not self.object_detected:
            self.no_depth_count += 1
            if self.no_depth_count > 10:
                self.update_state("NO_DEPTH_DATA")
            return

        self.no_depth_count = 0

        try:
            center_x = int(self.object_position[0])
            center_y = int(self.object_position[1])

            center_x = max(0, min(center_x, self.depth_image.shape[1] - 1))
            center_y = max(0, min(center_y, self.depth_image.shape[0] - 1))

            sample_size = 10
            depth_values = []

            for dx in range(-sample_size, sample_size + 1):
                for dy in range(-sample_size, sample_size + 1):
                    x = center_x + dx
                    y = center_y + dy
                    if (0 <= x < self.depth_image.shape[1] and
                            0 <= y < self.depth_image.shape[0]):
                        depth_val = self.depth_image[y, x]
                        if not np.isnan(depth_val) and 0.1 < depth_val < 10.0:
                            depth_values.append(float(depth_val))

            if depth_values:
                self.object_distance = float(np.median(depth_values))
                distance_msg = Float32()
                distance_msg.data = float(self.object_distance)
                self.distance_pub.publish(distance_msg)
            else:
                self.get_logger().warn('⚠️  No valid depth data at object position')
                if hasattr(self, 'last_bbox_area') and self.last_bbox_area > 0:
                    estimated_distance = max(0.5, min(3.0, 50000.0 / self.last_bbox_area))
                    self.object_distance = float(estimated_distance)
                    self.get_logger().info(f'📐 Estimated distance: {self.object_distance:.2f}m')
                else:
                    self.object_distance = 2.0

        except Exception as e:
            self.get_logger().error(f'❌ Distance calculation error: {e}')
            self.object_distance = 2.0

    def draw_yolo_detection(self, detection):
        """Draw YOLO detection on image."""
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(self.rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        self.last_bbox_area = (x2 - x1) * (y2 - y1)
        label = f'{self.target_object}: {detection["confidence"]:.2f}'
        if self.object_distance < float('inf'):
            label += f' | {self.object_distance:.2f}m'
        cv2.putText(self.rgb_image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def publish_detection_image(self):
        """Publish annotated image."""
        if self.rgb_image is None:
            return
        try:
            annotated_image = self.rgb_image.copy()
            h, w = annotated_image.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(annotated_image, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 2)
            cv2.line(annotated_image, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 2)

            status_text = (
                f"Target: {self.target_object} | Mode: {self.tracking_mode} | State: {self.current_state}"
            )
            cv2.putText(annotated_image, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.object_detected:
                detection_text = (
                    f"Confidence: {self.detection_confidence:.2f} | Distance: {self.object_distance:.2f}m"
                )
                cv2.putText(annotated_image, detection_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(
                    annotated_image,
                    (int(self.object_position[0]), int(self.object_position[1])),
                    10, (0, 0, 255), 3
                )

            detection_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.detection_image_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'❌ Image publishing error: {e}')

    # ─────────────────────────────────────────────
    # Safety
    # ─────────────────────────────────────────────

    def check_obstacles(self):
        """Check for obstacles using laser scan."""
        if self.laser_data is None:
            return
        ranges = np.array(self.laser_data.ranges)
        front_ranges = np.concatenate([ranges[-45:], ranges[:45]])
        valid_ranges = front_ranges[np.isfinite(front_ranges)]
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.emergency_stop = min_distance < self.min_obstacle_distance
            if self.emergency_stop:
                self.update_state("EMERGENCY_STOP")

    # ─────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────

    def control_loop(self):
        """Main control loop."""
        if self.emergency_stop:
            self.update_state("EMERGENCY_STOP")
            self.stop_robot()
            return

        if self.control_mode == "stopped":
            self.update_state("STOPPED")
            self.stop_robot()
            return

        if self.control_mode == "paused":
            self.update_state("PAUSED")
            self.stop_robot()
            return

        detection_msg = Bool()
        detection_msg.data = self.object_detected
        self.detection_pub.publish(detection_msg)

        if self.control_mode == "searching" or (
                self.control_mode == "following" and not self.object_detected):
            self.update_state("SEARCHING")
            self.search_behavior()
        elif self.control_mode == "following" and self.object_detected:
            if self.current_state == "SEARCHING":
                self.update_state("OBJECT_DETECTED")
            self.follow_behavior()

    def search_behavior(self):
        """Search for target object by rotating."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(twist)

    def follow_behavior(self):
        """Follow detected object."""
        if not self.object_detected or self.rgb_image is None:
            return

        # FIX 3: Invalid depth → clean mode transition, no thrashing.
        if self.object_distance == float('inf') or self.object_distance <= 0:
            self.get_logger().warn(f'⚠️  Invalid distance: {self.object_distance}')
            self.object_detected = False
            self.control_mode = "searching"
            self.update_state("NO_DEPTH_DATA")
            return

        twist = Twist()

        image_center_x = self.rgb_image.shape[1] // 2
        image_width = self.rgb_image.shape[1]

        angular_error = (self.object_position[0] - image_center_x) / (image_width / 2)
        distance_error = self.object_distance - self.target_distance

        is_aligned = abs(angular_error) < self.alignment_threshold
        at_target  = abs(distance_error) < self.dist_target_tolerance
        too_far    = distance_error >  self.dist_target_tolerance
        too_close  = distance_error < -self.dist_target_tolerance

        if at_target and is_aligned:
            self.update_state("AT_TARGET")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("🎯 Target reached, waiting for next command")
            # Hold position until a new target command arrives
            self.control_mode = "stopped"

        elif not is_aligned:
            self.update_state("ALIGNING")
            twist.angular.z = -angular_error * self.angular_gain
            twist.linear.x = 0.0

        elif too_far and is_aligned:
            self.update_state("APPROACHING")
            twist.linear.x = min(distance_error * self.linear_gain, self.max_linear_speed)
            twist.angular.z = -angular_error * self.angular_gain * 0.3

        elif too_close and is_aligned:
            self.update_state("REVERSING")
            twist.linear.x = max(distance_error * self.linear_gain, -self.max_linear_speed * 0.5)
            twist.angular.z = 0.0

        # Apply velocity limits
        twist.linear.x  = max(min(twist.linear.x,  self.max_linear_speed),  -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)

        self.cmd_vel_pub.publish(twist)

        # Periodic status log (every 2 s in same state)
        if time.time() - self.state_change_time > 2.0:
            self.get_logger().info(
                f'📊 Status: State={self.current_state}, '
                f'Distance={self.object_distance:.2f}m (target={self.target_distance:.2f}m), '
                f'Align_err={angular_error:.2f}, '
                f'Speed: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}'
            )
            self.state_change_time = time.time()

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedYOLOTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()