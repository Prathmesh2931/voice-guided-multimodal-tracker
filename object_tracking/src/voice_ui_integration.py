#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import sounddevice as sd
import soundfile as sf
import subprocess
from threading import Thread
from dotenv import load_dotenv
from groq import Groq
import re
import numpy as np
from pathlib import Path


class VoiceObjectController(Node):
    def __init__(self):
        super().__init__('voice_object_controller')

        # Load environment variables
        current_dir = Path(__file__).resolve()
        for parent in current_dir.parents:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
        else:
            print("⚠️  .env file not found anywhere above this script.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.get_logger().error("❌ GROQ_API_KEY not found in environment! Check your .env file.")
            raise ValueError("Missing GROQ_API_KEY in .env")

        self.client = Groq(api_key=api_key)

        # ── NEW: trigger subscriber ─────────────────────────────────────────
        self.is_listening = False
        self.listen_trigger_sub = self.create_subscription(
            Bool,
            '/start_listening',
            self.listen_callback,
            10
        )
        # ───────────────────────────────────────────────────────────────────

        # Publishers (unchanged)
        self.target_object_pub = self.create_publisher(String, '/target_object', 10)
        self.control_mode_pub  = self.create_publisher(String, '/control_mode', 10)
        self.search_mode_pub   = self.create_publisher(Bool,   '/search_mode',  10)

        # Command mappings (unchanged — keeping all original dicts)
        self.direct_object_commands = {
            "follow person": "person",  "track person": "person",  "find person": "person",
            "follow human":  "person",  "track human":  "person",  "find human":  "person",
            "follow chair":  "chair",   "track chair":  "chair",   "find chair":  "chair",
            "follow bottle": "bottle",  "track bottle": "bottle",  "find bottle": "bottle",
            "follow laptop": "laptop",  "track laptop": "laptop",  "find laptop": "laptop",
            "follow phone":  "cell phone", "track phone": "cell phone", "find phone": "cell phone",
            "follow car":    "car",     "track car":    "car",     "find car":    "car",
            "follow dog":    "dog",     "track dog":    "dog",     "find dog":    "dog",
            "follow cat":    "cat",     "track cat":    "cat",     "find cat":    "cat",
            "follow cup":    "cup",     "track cup":    "cup",     "find cup":    "cup",
            "follow book":   "book",    "track book":   "book",    "find book":   "book",
            "follow bag":    "backpack","track bag":    "backpack","find bag":    "backpack",
            "follow backpack":"backpack","track backpack":"backpack","find backpack":"backpack",
            "follow red box":    "red_object",  "track red box":    "red_object",  "find red box":    "red_object",
            "follow red object": "red_object",  "track red object": "red_object",  "find red object": "red_object",
            "follow blue box":   "blue_object", "track blue box":   "blue_object", "find blue box":   "blue_object",
            "follow green box":  "green_object","track green box":  "green_object","find green box":  "green_object",
        }

        self.control_commands = {
            "stop": "stop", "halt": "stop", "pause": "pause",
            "resume": "resume", "search": "search",
            "look around": "search", "find something": "search", "scan area": "search",
        }

        self.single_word_objects = {
            "person": "person", "human": "person", "chair": "chair",
            "bottle": "bottle", "laptop": "laptop", "phone": "cell phone",
            "car": "car", "dog": "dog", "cat": "cat",
            "cup": "cup", "book": "book", "bag": "backpack", "backpack": "backpack",
        }

        self.get_logger().info("Voice Object Controller initialized")
        self.get_logger().info("Waiting for UI trigger on /start_listening...")

        self.start_voice_recognition()

    # ── NEW: trigger callback ───────────────────────────────────────────────
    def listen_callback(self, msg):
        self.is_listening = msg.data
        state = "🎤 Listening triggered from UI" if msg.data else "⏹ Listening stopped from UI"
        self.get_logger().info(state)
    # ───────────────────────────────────────────────────────────────────────

    def start_voice_recognition(self):
        self.voice_thread = Thread(target=self.voice_recognition_loop, daemon=True)
        self.voice_thread.start()

    def voice_recognition_loop(self):
        try:
            while rclpy.ok():
                # ── MODIFIED: only process when UI has triggered listening ──
                if not self.is_listening:
                    continue
                # ────────────────────────────────────────────────────────────

                try:
                    audio_file = self.record_audio(duration=4)
                    transcript = self.transcribe_audio(audio_file)
                    transcript = self.clean_text(transcript)

                    if transcript and len(transcript.strip()) > 0:
                        self.get_logger().info(f"Heard: '{transcript}'")
                        self.parse_and_execute_command(transcript)

                except Exception as e:
                    self.get_logger().error(f"Voice recognition error: {e}")

        except KeyboardInterrupt:
            self.get_logger().info("Voice recognition stopped")

    def record_audio(self, duration=4, sample_rate=16000, output_file="/tmp/voice_input.wav"):
        try:
            self.get_logger().debug("Listening...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            if np.max(np.abs(audio)) < 500:
                return None
            sf.write(output_file, audio, sample_rate)
            return output_file
        except Exception as e:
            self.get_logger().error(f"Audio recording error: {e}")
            return None

    def transcribe_audio(self, file_path):
        if not file_path:
            return ""
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(file_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="en",
                )
            return transcription.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return ""

    def clean_text(self, text):
        if not text:
            return ""
        return text.lower().strip().replace('.','').replace(',','').replace('!','').replace('?','')

    def parse_and_execute_command(self, command):
        if self.try_direct_pattern_matching(command):
            return
        try:
            self.try_llm_parsing(command)
        except Exception as e:
            self.get_logger().error(f"LLM parsing failed: {e}")
            self.try_keyword_matching(command)

    def try_direct_pattern_matching(self, command):
        if command in self.direct_object_commands:
            self.execute_object_tracking(self.direct_object_commands[command])
            return True
        if command in self.control_commands:
            self.execute_control_command(self.control_commands[command])
            return True
        for action in ["follow", "track", "find"]:
            if command.startswith(action + " "):
                object_part = command.replace(action + " ", "").strip()
                if object_part in self.single_word_objects:
                    self.execute_object_tracking(self.single_word_objects[object_part])
                    return True
                color_command = action + " " + object_part
                if color_command in self.direct_object_commands:
                    obj = self.direct_object_commands[color_command]
                    if "object" in obj:
                        self.execute_color_object_tracking(obj)
                    else:
                        self.execute_object_tracking(obj)
                    return True
        return False

    def try_llm_parsing(self, command):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a command parser for robot object tracking.
                    Available object commands: {list(self.direct_object_commands.keys())}
                    Available control commands: {list(self.control_commands.keys())}
                    Parse the user's command and return ONLY ONE of these formats:
                    - "OBJECT:[object_name]" for tracking commands
                    - "CONTROL:[action]" for control commands
                    - "COLOR_OBJECT:[color]_object" for colored objects
                    - "INVALID" if command doesn't match"""
                },
                {"role": "user", "content": f"Parse this command: '{command}'"}
            ],
            model="llama-3.1-8b-instant",
        )
        parsed_result = chat_completion.choices[0].message.content.strip().strip('"').strip("'")
        self.get_logger().info(f"LLM parsed command: {parsed_result}")

        if parsed_result.startswith("OBJECT:"):
            self.execute_object_tracking(parsed_result.replace("OBJECT:", ""))
        elif parsed_result.startswith("COLOR_OBJECT:"):
            self.execute_color_object_tracking(parsed_result.replace("COLOR_OBJECT:", ""))
        elif parsed_result.startswith("CONTROL:"):
            self.execute_control_command(parsed_result.replace("CONTROL:", ""))
        else:
            raise Exception("LLM parsing failed")

    def try_keyword_matching(self, command):
        self.get_logger().info("Trying keyword matching as final fallback")
        if any(w in command for w in ["stop", "halt"]):
            self.execute_control_command("stop"); return
        if any(w in command for w in ["search", "look", "scan"]):
            self.execute_control_command("search"); return
        if "pause"  in command: self.execute_control_command("pause");  return
        if "resume" in command: self.execute_control_command("resume"); return
        for obj_key, obj_value in self.single_word_objects.items():
            if obj_key in command and any(a in command for a in ["follow","track","find"]):
                self.execute_object_tracking(obj_value); return
        if "red"   in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("red_object");   return
        if "blue"  in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("blue_object");  return
        if "green" in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("green_object"); return
        self.get_logger().warn(f"Could not understand command: '{command}'")

    def execute_object_tracking(self, object_name):
        self.get_logger().info(f"Starting to track: {object_name}")
        msg = String(); msg.data = object_name;     self.target_object_pub.publish(msg)
        msg = String(); msg.data = "following";     self.control_mode_pub.publish(msg)
        msg = Bool();   msg.data = False;           self.search_mode_pub.publish(msg)

    def execute_color_object_tracking(self, color_object):
        self.get_logger().info(f"Starting to track: {color_object}")
        msg = String(); msg.data = color_object;    self.target_object_pub.publish(msg)
        msg = String(); msg.data = "color_tracking";self.control_mode_pub.publish(msg)
        msg = Bool();   msg.data = False;           self.search_mode_pub.publish(msg)

    def execute_control_command(self, action):
        self.get_logger().info(f"Executing control action: {action}")
        if action in ["stop", "halt"]:
            msg = String(); msg.data = "stopped";  self.control_mode_pub.publish(msg)
        elif action == "pause":
            msg = String(); msg.data = "paused";   self.control_mode_pub.publish(msg)
        elif action == "resume":
            msg = String(); msg.data = "following"; self.control_mode_pub.publish(msg)
        elif action == "search":
            msg = Bool();   msg.data = True;        self.search_mode_pub.publish(msg)
            msg = String(); msg.data = "searching"; self.control_mode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceObjectController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()