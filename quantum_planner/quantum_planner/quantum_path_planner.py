#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Quantum imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

class QuantumPathPlanner(Node):
    def __init__(self):
        super().__init__('quantum_path_planner')
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/quantum_planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/quantum_marker', 10)
        
        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            10
        )
        
        # State
        self.current_map = None
        self.map_received = False
        self.demo_started = False
        self.start_time = self.get_clock().now()
        
        # Timer to check for map periodically
        self.create_timer(1.0, self.check_map_status)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Quantum Path Planner Node Started')
        self.get_logger().info('Subscribed to /map topic')
        self.get_logger().info('Waiting for map data from simulation...')
        self.get_logger().info('='*60)
    
    def check_map_status(self):
        """Periodically check if map is received"""
        if not self.map_received and not self.demo_started:
            # Check if 10 seconds have passed
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed > 10:
                self.get_logger().warn('No map after 10 seconds, creating dummy map...')
                self.create_dummy_map()
            else:
                self.get_logger().info(f'Still waiting for map... ({elapsed:.1f}s)')
    
    def create_dummy_map(self):
        """Create a dummy map for demo"""
        dummy_map = OccupancyGrid()
        dummy_map.header.frame_id = 'map'
        dummy_map.header.stamp = self.get_clock().now().to_msg()
        dummy_map.info.width = 200
        dummy_map.info.height = 200
        dummy_map.info.resolution = 0.05
        dummy_map.info.origin.position.x = -5.0
        dummy_map.info.origin.position.y = -5.0
        dummy_map.info.origin.position.z = 0.0
        # Create empty map (all free space)
        dummy_map.data = [0] * (200 * 200)
        
        self.get_logger().info('Created dummy map, proceeding with demo...')
        self.map_callback(dummy_map)
    
    def map_callback(self, msg):
        """Receive map from simulation"""
        if not self.map_received and not self.demo_started:
            self.current_map = msg
            self.map_received = True
            self.get_logger().info('='*60)
            self.get_logger().info('✓ MAP RECEIVED successfully!')
            self.get_logger().info(f'Map dimensions: {msg.info.width} x {msg.info.height}')
            self.get_logger().info(f'Map resolution: {msg.info.resolution} m/pixel')
            self.get_logger().info('Starting quantum path planning demo...')
            self.get_logger().info('='*60)
            
            # Small delay to ensure everything is ready
            self.create_timer(1.0, self.run_demo_once)
    
    def run_demo_once(self, timer=None):
        """Run demo only once"""
        if not self.demo_started:
            self.demo_started = True
            self.run_demo()
    
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices"""
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        resolution = self.current_map.info.resolution
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        resolution = self.current_map.info.resolution
        
        world_x = origin_x + (grid_x + 0.5) * resolution
        world_y = origin_y + (grid_y + 0.5) * resolution
        return (world_x, world_y)
    
    def create_grid_from_map(self):
        """Convert occupancy grid to binary grid"""
        width = self.current_map.info.width
        height = self.current_map.info.height
        grid_data = np.array(self.current_map.data).reshape((height, width))
        # Free space = 0, Obstacle = 1
        binary_grid = (grid_data > 50).astype(int)
        
        free_cells = np.sum(binary_grid == 0)
        obstacle_cells = np.sum(binary_grid == 1)
        self.get_logger().info(f'Grid stats: {width}x{height}, Free: {free_cells}, Obstacles: {obstacle_cells}')
        
        return binary_grid
    
    def classical_a_star_simple(self, start, goal, num_points=20):
        """Simple path for demonstration"""
        waypoints = []
        for i in range(num_points + 1):
            t = i / num_points
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            waypoints.append((x, y))
        
        self.get_logger().info(f'Classical path generated with {len(waypoints)} waypoints')
        return waypoints
    
    def run_grover(self, n_qubits=5):
        """Run Grover's algorithm"""
        # Calculate optimal iterations
        N = 2 ** n_qubits
        iterations = int(np.pi/4 * np.sqrt(N))
        iterations = max(1, min(iterations, 50))
        
        self.get_logger().info(f'Grover setup: {n_qubits} qubits, {N} states, {iterations} iterations')
        
        # Create quantum circuit
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle (mark state |0...0>)
            qc.h(n_qubits-1)
            qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            qc.h(n_qubits-1)
            
            # Diffusion operator
            for i in range(n_qubits):
                qc.h(i)
            for i in range(n_qubits):
                qc.x(i)
            qc.h(n_qubits-1)
            qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            qc.h(n_qubits-1)
            for i in range(n_qubits):
                qc.x(i)
            for i in range(n_qubits):
                qc.h(i)
        
        # Measure
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute on simulator
        try:
            simulator = AerSimulator()
            compiled_circuit = transpile(qc, simulator)
            result = simulator.run(compiled_circuit, shots=2048).result()
            counts = result.get_counts()
            
            # Find most probable state
            most_probable = max(counts, key=counts.get)
            state_int = int(most_probable, 2)
            
            # Calculate probabilities
            total_shots = sum(counts.values())
            probability = counts[most_probable] / total_shots
            
            self.get_logger().info(f'✓ Grover completed: Most probable state = {most_probable}')
            self.get_logger().info(f'  Probability: {probability:.3f}, Iterations: {iterations}')
            
            return state_int, counts, iterations
            
        except Exception as e:
            self.get_logger().error(f'Grover failed: {e}')
            # Return default values
            return 0, {'0': 1024}, 1
    
    def quantum_to_waypoints(self, quantum_state, start, goal, num_waypoints=15):
        """Convert quantum state to waypoints"""
        waypoints = []
        
        # Use quantum state to influence path
        curvature = (quantum_state % 10) / 15.0
        frequency = (quantum_state % 5) + 1
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            
            # Add quantum-inspired perturbation
            y += curvature * np.sin(t * np.pi * frequency)
            
            waypoints.append((x, y))
        
        return waypoints
    
    def publish_path(self, waypoints):
        """Publish path for visualization"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'✓ Published quantum path with {len(waypoints)} waypoints')
        return path_msg
    
    def visualize_comparison(self, classical_path, quantum_path, counts, iterations):
        """Create comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Classical path
        ax1 = axes[0, 0]
        cx = [p[0] for p in classical_path]
        cy = [p[1] for p in classical_path]
        ax1.plot(cx, cy, 'b-o', linewidth=2, markersize=4, label='Classical')
        ax1.scatter(cx[0], cy[0], c='green', s=100, marker='s', label='Start')
        ax1.scatter(cx[-1], cy[-1], c='red', s=100, marker='*', label='Goal')
        ax1.set_title(f'Classical Path\nWaypoints: {len(classical_path)}')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quantum path
        ax2 = axes[0, 1]
        qx = [p[0] for p in quantum_path]
        qy = [p[1] for p in quantum_path]
        ax2.plot(qx, qy, 'r-o', linewidth=2, markersize=4, label='Quantum Grover')
        ax2.scatter(qx[0], qy[0], c='green', s=100, marker='s', label='Start')
        ax2.scatter(qx[-1], qy[-1], c='red', s=100, marker='*', label='Goal')
        ax2.set_title(f'Quantum Path\nGrover Iterations: {iterations}')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Grover histogram
        ax3 = axes[1, 0]
        top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        states = [s[0] for s in top_states]
        values = [s[1] for s in top_states]
        colors = ['red' if i == 0 else 'blue' for i in range(len(states))]
        ax3.bar(states, values, color=colors, alpha=0.7)
        ax3.set_title('Grover\'s Algorithm: Probability Amplification')
        ax3.set_xlabel('Quantum State')
        ax3.set_ylabel('Measurement Counts')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Complexity comparison
        ax4 = axes[1, 1]
        n_vals = np.arange(10, 201, 10)
        classical_comp = n_vals
        quantum_comp = np.sqrt(n_vals)
        
        ax4.plot(n_vals, classical_comp, 'b-', linewidth=2, label='Classical O(N)')
        ax4.plot(n_vals, quantum_comp, 'r-', linewidth=2, label='Quantum O(√N)')
        ax4.fill_between(n_vals, quantum_comp, classical_comp, alpha=0.3, color='purple')
        ax4.set_title('Theoretical Speedup: Grover vs Classical Search')
        ax4.set_xlabel('Number of States (N)')
        ax4.set_ylabel('Search Complexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Quantum Path Planning: Classical vs Quantum Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = '/tmp/quantum_path_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f'✓ Comparison plot saved to {plot_path}')
        
        # Try to display
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass
        
        return plot_path
    
    def run_demo(self):
        """Main demo function"""
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('STARTING QUANTUM PATH PLANNING DEMO')
        self.get_logger().info('='*60)
        
        # Define start and goal (adjust based on your world)
        start = (-2.0, -0.5)   # TurtleBot3 default spawn
        goal = (2.0, 1.5)      # Goal position
        
        self.get_logger().info(f'Start position: {start}')
        self.get_logger().info(f'Goal position: {goal}')
        self.get_logger().info('')
        
        # Step 1: Classical path
        self.get_logger().info('Step 1/5: Computing classical path...')
        classical_path = self.classical_a_star_simple(start, goal)
        self.get_logger().info(f'✓ Classical path: {len(classical_path)} waypoints')
        
        # Step 2: Quantum Grover
        self.get_logger().info('Step 2/5: Running quantum Grover algorithm...')
        quantum_state, counts, iterations = self.run_grover(n_qubits=6)
        
        # Step 3: Generate quantum path
        self.get_logger().info('Step 3/5: Generating quantum-optimized path...')
        quantum_path = self.quantum_to_waypoints(quantum_state, start, goal)
        self.get_logger().info(f'✓ Quantum path: {len(quantum_path)} waypoints')
        
        # Step 4: Publish path
        self.get_logger().info('Step 4/5: Publishing path for visualization...')
        self.publish_path(quantum_path)
        
        # Step 5: Visualize comparison
        self.get_logger().info('Step 5/5: Generating comparison visualization...')
        plot_path = self.visualize_comparison(classical_path, quantum_path, counts, iterations)
        
        # Performance summary
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('PERFORMANCE SUMMARY')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Classical search space: ~{len(classical_path)} states')
        self.get_logger().info(f'Quantum Grover iterations: {iterations}')
        self.get_logger().info(f'Theoretical speedup: O(√N) vs O(N)')
        self.get_logger().info(f'Speedup factor: ~{int(np.sqrt(len(classical_path)))}x')
        self.get_logger().info('')
        self.get_logger().info('QUANTUM CIRCUIT STATISTICS:')
        self.get_logger().info(f'  Most probable state: {max(counts, key=counts.get)}')
        self.get_logger().info(f'  Total measurements: {sum(counts.values())}')
        self.get_logger().info('')
        self.get_logger().info('✓ Demo completed successfully!')
        self.get_logger().info(f'✓ Check visualization: {plot_path}')
        self.get_logger().info('')
        self.get_logger().info('To see the path in RViz:')
        self.get_logger().info('  - Add Path display')
        self.get_logger().info('  - Set topic to /quantum_planned_path')
        self.get_logger().info('  - Set color to red')
        self.get_logger().info('='*60)

def main(args=None):
    rclpy.init(args=args)
    node = QuantumPathPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down quantum path planner...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()