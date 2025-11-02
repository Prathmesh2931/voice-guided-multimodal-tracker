#!/usr/bin/env python3

def generate_chair_sdf(scale=1.0):
    # Calculate all scaled dimensions
    seat_size = 0.45 * scale
    seat_height = 0.45 * scale
    seat_thickness = 0.05 * scale
    backrest_width = 0.45 * scale
    backrest_height = 0.6 * scale
    backrest_thickness = 0.05 * scale
    backrest_offset = -0.2 * scale
    backrest_center_height = 0.75 * scale
    leg_radius = 0.025 * scale
    leg_length = 0.45 * scale
    leg_offset = 0.175 * scale
    leg_center_height = 0.225 * scale
    seat_mass = 5.0 * (scale ** 3)
    backrest_mass = 3.0 * (scale ** 3)
    leg_mass = 1.0 * (scale ** 3)
    
    sdf_content = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="chair">
    <static>false</static>
    <pose>0 0 0 0 0 0</pose>
    
    <!-- Chair Seat -->
    <link name="seat">
      <pose>0 0 {seat_height} 0 0 0</pose>
      <inertial>
        <mass>{seat_mass}</mass>
        <inertia>
          <ixx>{0.083 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.083 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.083 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="seat_collision">
        <geometry>
          <box>
            <size>{seat_size} {seat_size} {seat_thickness}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>
              <mu2>0.8</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="seat_visual">
        <geometry>
          <box>
            <size>{seat_size} {seat_size} {seat_thickness}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
          <diffuse>0.6 0.4 0.2 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Chair Backrest -->
    <link name="backrest">
      <pose>0 {backrest_offset} {backrest_center_height} 0 0 0</pose>
      <inertial>
        <mass>{backrest_mass}</mass>
        <inertia>
          <ixx>{0.25 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.05 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.25 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="backrest_collision">
        <geometry>
          <box>
            <size>{backrest_width} {backrest_thickness} {backrest_height}</size>
          </box>
        </geometry>
      </collision>
      <visual name="backrest_visual">
        <geometry>
          <box>
            <size>{backrest_width} {backrest_thickness} {backrest_height}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
          <diffuse>0.6 0.4 0.2 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Front Left Leg -->
    <link name="front_left_leg">
      <pose>{leg_offset} {leg_offset} {leg_center_height} 0 0 0</pose>
      <inertial>
        <mass>{leg_mass}</mass>
        <inertia>
          <ixx>{0.015 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.015 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.001 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="front_left_leg_collision">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="front_left_leg_visual">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.3 0.1 1</ambient>
          <diffuse>0.5 0.3 0.1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Front Right Leg -->
    <link name="front_right_leg">
      <pose>{-leg_offset} {leg_offset} {leg_center_height} 0 0 0</pose>
      <inertial>
        <mass>{leg_mass}</mass>
        <inertia>
          <ixx>{0.015 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.015 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.001 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="front_right_leg_collision">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="front_right_leg_visual">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.3 0.1 1</ambient>
          <diffuse>0.5 0.3 0.1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Back Left Leg -->
    <link name="back_left_leg">
      <pose>{leg_offset} {-leg_offset} {leg_center_height} 0 0 0</pose>
      <inertial>
        <mass>{leg_mass}</mass>
        <inertia>
          <ixx>{0.015 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.015 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.001 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="back_left_leg_collision">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="back_left_leg_visual">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.3 0.1 1</ambient>
          <diffuse>0.5 0.3 0.1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Back Right Leg -->
    <link name="back_right_leg">
      <pose>{-leg_offset} {-leg_offset} {leg_center_height} 0 0 0</pose>
      <inertial>
        <mass>{leg_mass}</mass>
        <inertia>
          <ixx>{0.015 * scale}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{0.015 * scale}</iyy>
          <iyz>0.0</iyz>
          <izz>{0.001 * scale}</izz>
        </inertia>
      </inertial>
      <collision name="back_right_leg_collision">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="back_right_leg_visual">
        <geometry>
          <cylinder>
            <radius>{leg_radius}</radius>
            <length>{leg_length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.3 0.1 1</ambient>
          <diffuse>0.5 0.3 0.1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Fixed Joints -->
    <joint name="seat_to_front_left_leg" type="fixed">
      <parent>seat</parent>
      <child>front_left_leg</child>
    </joint>

    <joint name="seat_to_front_right_leg" type="fixed">
      <parent>seat</parent>
      <child>front_right_leg</child>
    </joint>

    <joint name="seat_to_back_left_leg" type="fixed">
      <parent>seat</parent>
      <child>back_left_leg</child>
    </joint>

    <joint name="seat_to_back_right_leg" type="fixed">
      <parent>seat</parent>
      <child>back_right_leg</child>
    </joint>

    <joint name="seat_to_backrest" type="fixed">
      <parent>seat</parent>
      <child>backrest</child>
    </joint>

  </model>
</sdf>'''
    
    return sdf_content

if __name__ == "__main__":
    import sys
    
    # Get scale from command line or use default
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    
    sdf_content = generate_chair_sdf(scale)
    print(sdf_content)