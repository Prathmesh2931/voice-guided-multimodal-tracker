
---

# 🎤 Voice-Controlled YOLO Object Tracking (ROS2)

## 🚀 Run

Open **2 terminals**

### Terminal 1 (Simulation + Tracking)

```bash
cd ~/ml
source install/setup.bash
ros2 launch turtlebot3 yolo_object_follow.launch.py
```

### Terminal 2 (Voice Control)

```bash
cd ~/ml
source install/setup.bash
ros2 run object_tracking voice_cmd_interface.py
```

---

## 🎤 Commands

```text
Follow chair
Follow person
Follow red object
Follow blue object
Stop
```

---

## ⚙️ Requirements

```bash
sudo apt install ros-humble-desktop ros-humble-turtlebot3*
pip install ultralytics opencv-python torch numpy
pip install sounddevice soundfile groq python-dotenv
```

```bash
echo 'export TURTLEBOT3_MODEL=waffle' >> ~/.bashrc
source ~/.bashrc
```

---

## ⚠️ Notes

* Use **"following" mode internally** (not `color_tracking`)
* If robot spins → object not detected
* Check camera:

```bash
rqt_image_view
```

---

## ✅ Done

```bash
ros2 launch turtlebot3 yolo_object_follow.launch.py
ros2 run object_tracking voice_cmd_interface.py
```

