#!/usr/bin/env python3
"""
RL Navigation Node — Turtlebot3 + Gazebo Fortress + ROS2
FIXED: Proper reset + safety recovery + better initial positioning
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import threading
import time

try:
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[RL] stable-baselines3 not found")


class TurtlebotNavEnv(gym.Env if SB3_AVAILABLE else object):
    N_BINS = 24
    MIN_RANGE = 0.12
    MAX_RANGE = 3.5
    SAFE_DIST = 0.5
    LINEAR_VEL = 0.15
    ANGULAR_VEL = 0.6

    def __init__(self, ros_node: "RLAgentNode"):
        if SB3_AVAILABLE:
            super().__init__()
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.N_BINS,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)

        self.node = ros_node
        self.step_count = 0
        self.max_steps = 300
        self.episode = 0
        self.total_reward = 0.0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.total_reward = 0.0
        self.episode += 1
        
        self.node.get_logger().info(f"[ENV] Episode {self.episode} - Resetting robot position...")
        
        # CRITICAL: Stop the robot first
        self._stop_robot()
        time.sleep(0.3)
        
        # Try to get to a safe position by backing up and rotating
        self._recover_from_collision()
        
        # Wait for Gazebo to settle
        time.sleep(1.0)
        
        obs = self._get_obs()
        
        # Check if reset position is safe
        distances = obs * self.MAX_RANGE
        if distances.min() < self.MIN_RANGE * 2:
            self.node.get_logger().warn(f"[ENV] Reset position unsafe! Min dist: {distances.min():.2f}m")
            # Try one more recovery
            self._emergency_recovery()
            obs = self._get_obs()
        
        return obs, {}

    def _recover_from_collision(self):
        """Back up and rotate to find clear space"""
        self.node.get_logger().info("[ENV] Recovery: Backing up...")
        
        # Back up for 1 second
        cmd = Twist()
        cmd.linear.x = -0.2  # Backward
        for _ in range(10):
            self.node.cmd_pub.publish(cmd)
            time.sleep(0.1)
        
        # Rotate to find clear path
        self.node.get_logger().info("[ENV] Recovery: Rotating to find clear space...")
        cmd = Twist()
        cmd.angular.z = 0.5
        for _ in range(20):  # Rotate for 2 seconds
            self.node.cmd_pub.publish(cmd)
            time.sleep(0.1)
            
            # Check if we found clear space
            obs = self._get_obs()
            if obs.min() * self.MAX_RANGE > 0.3:
                break
        
        self._stop_robot()
        
    def _emergency_recovery(self):
        """Emergency: rotate in place until finding clear direction"""
        self.node.get_logger().warn("[ENV] EMERGENCY RECOVERY - Searching for clear path...")
        
        for angle in range(8):  # Try 8 different directions
            cmd = Twist()
            cmd.angular.z = 0.8
            for _ in range(10):
                self.node.cmd_pub.publish(cmd)
                time.sleep(0.1)
            
            self._stop_robot()
            time.sleep(0.2)
            
            obs = self._get_obs()
            min_dist = obs.min() * self.MAX_RANGE
            if min_dist > 0.3:
                self.node.get_logger().info(f"[ENV] Found clear direction! Dist: {min_dist:.2f}m")
                break

    def step(self, action: int):
        self._apply_action(action)
        time.sleep(0.1)

        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs, action)
        truncated = self.step_count >= self.max_steps
        self.total_reward += reward
        self.step_count += 1

        # Early termination if robot is stuck in bad state
        if obs.min() * self.MAX_RANGE < self.MIN_RANGE:
            terminated = True
            reward -= 10.0

        if terminated or truncated:
            self.node.get_logger().info(
                f"[ENV] Episode {self.episode} ended | "
                f"steps={self.step_count} | total_reward={self.total_reward:.2f}"
            )
            self._stop_robot()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        scan = self.node.latest_scan
        if scan is None or len(scan.ranges) == 0:
            return np.ones(self.N_BINS, dtype=np.float32)

        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.MAX_RANGE)
        ranges = np.clip(ranges, 0.0, self.MAX_RANGE)

        indices = np.linspace(0, len(ranges) - 1, self.N_BINS, dtype=int)
        obs = ranges[indices] / self.MAX_RANGE
        return obs.astype(np.float32)

    def _get_sector_distances(self, distances):
        """Calculate 45° sector minima"""
        front_bins = [22, 23, 0, 1, 2]
        front_left_bins = [3, 4, 5]
        left_bins = [6, 7, 8]
        front_right_bins = [19, 20, 21]
        right_bins = [15, 16, 17]
        
        return {
            'front': distances[front_bins].min(),
            'front_left': distances[front_left_bins].min(),
            'left': distances[left_bins].min(),
            'front_right': distances[front_right_bins].min(),
            'right': distances[right_bins].min(),
        }

    def _compute_reward(self, obs: np.ndarray, action: int):
        distances = obs * self.MAX_RANGE
        sectors = self._get_sector_distances(distances)
        
        min_dist = distances.min()
        
        # Severe collision penalty
        if min_dist < self.MIN_RANGE:
            return -20.0, True
        
        reward = 0.0
        
        # Survival bonus (small positive for each step without collision)
        reward += 0.1
        
        # Forward progress
        if action == 0:
            reward += 0.5
            if sectors['front'] > self.SAFE_DIST:
                reward += 0.3  # Extra for clear path
        else:
            reward -= 0.05  # Small turning penalty
        
        # Obstacle avoidance
        if sectors['front'] < self.SAFE_DIST:
            # Penalty based on proximity
            reward -= (self.SAFE_DIST - sectors['front']) * 2.0
            
            # Reward correct turning
            left_clear = min(sectors['front_left'], sectors['left'])
            right_clear = min(sectors['front_right'], sectors['right'])
            
            if left_clear > right_clear and action == 1:  # Turn left when left is clearer
                reward += 0.5
            elif right_clear > left_clear and action == 2:  # Turn right when right is clearer
                reward += 0.5
        
        # Exploration bonus (encourage moving away from walls)
        if min_dist > 0.5:
            reward += 0.2
        
        return reward, False

    def _apply_action(self, action: int):
        cmd = Twist()
        if action == 0:  # Forward
            cmd.linear.x = self.LINEAR_VEL
        elif action == 1:  # Rotate Left
            cmd.angular.z = self.ANGULAR_VEL
        else:  # Rotate Right
            cmd.angular.z = -self.ANGULAR_VEL
        self.node.cmd_pub.publish(cmd)

    def _stop_robot(self):
        self.node.cmd_pub.publish(Twist())


class RLAgentNode(Node):
    def __init__(self):
        super().__init__("rl_agent_node")
        self.get_logger().info("RL Agent Node initialised")

        self.latest_scan = None
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg


def main():
    rclpy.init()
    node = RLAgentNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    node.get_logger().info("Waiting for /scan data...")
    while node.latest_scan is None:
        time.sleep(0.1)
    node.get_logger().info("/scan received — starting")

    env = TurtlebotNavEnv(ros_node=node)

    if SB3_AVAILABLE:
        node.get_logger().info("Training with PPO (200K steps) - USING CPU for stability...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,  # Increased for better learning
            batch_size=64,
            n_epochs=10,
            device='cpu',  # Force CPU to avoid GPU warnings
        )
        model.learn(total_timesteps=200_000)
        model.save("turtlebot3_ppo_policy")
        node.get_logger().info("Model saved")
    else:
        node.get_logger().info("Running heuristic policy...")
        for episode in range(20):
            obs, _ = env.reset()
            done = False
            episode_steps = 0
            
            while not done and episode_steps < 500:
                distances = obs * env.MAX_RANGE
                sectors = env._get_sector_distances(distances)
                
                # Simple reactive policy
                if sectors['front'] < 0.3:
                    left_clear = min(sectors['front_left'], sectors['left'])
                    right_clear = min(sectors['front_right'], sectors['right'])
                    action = 1 if left_clear > right_clear else 2
                elif sectors['left'] < 0.25:
                    action = 2
                elif sectors['right'] < 0.25:
                    action = 1
                else:
                    action = 0
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_steps += 1
                
                if episode_steps % 50 == 0:
                    node.get_logger().info(
                        f"Ep{episode+1} Step{episode_steps}: "
                        f"F={sectors['front']:.2f} L={sectors['left']:.2f} R={sectors['right']:.2f}"
                    )

    node.get_logger().info("RL loop complete")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()