# =========================================================================
#  Be More Agent (Hailo Optimized) 🤖
#  Simplified for Pi 5 + Hailo-10H + USB Mic
# =========================================================================

import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import random
import re
import traceback
import atexit
import datetime
import urllib.request
import urllib.error

import numpy as np

# AI Engines
from openwakeword.model import Model

# Core modules
from core.llm import Brain, init_llm, is_llm_ready, _get_llm
from core.npu import _release_llm, reload_after_vlm
from core.tts import init_audio, shutdown_audio, get_player, get_synthesizer, clean_text_for_speech
from core.stt import transcribe_audio, init_stt
from core.config import WAKE_WORD_MODEL, ALSA_DEVICE, FOLLOWUP_ENABLED, LANGUAGE, t
from core.audio_input import wait_for_wakeword, record_until_silence
from core.dispatch import dispatch_stream
from core.screensaver import screensaver_loop
from core.meter import MicMeter
from core.bubble import ThoughtBubble
from core.log import bmo_print, setup_logging

setup_logging()

# =========================================================================
# 1. HARDWARE CONFIGURATION
# =========================================================================

# =========================================================================
# 2. GUI & STATE
# =========================================================================

class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    CAPTURING = "capturing"
    WARMUP = "warmup"
    DISPLAY_IMAGE = "display_image"
    SCREENSAVER = "screensaver"
    # New Expressions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    DIZZY = "dizzy"
    CHEEKY = "cheeky"
    HEART = "heart"
    STARRY_EYED = "starry_eyed"
    CONFUSED = "confused"
    SHHH = "shhh"
    JAMMING = "jamming"
    FOOTBALL = "football"
    DETECTIVE = "detective"
    SIR_MANO = "sir_mano"
    LOW_BATTERY = "low_battery"
    BEE = "bee"

class BotGUI:

    BG_WIDTH, BG_HEIGHT = 800, 480
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True) 
        master.configure(cursor='none') # Hide cursor for kiosk display
        master.bind('<Escape>', self.exit_fullscreen)
        master.bind('<Button-1>', self.mute_bmo)
        
        # Events
        self.stop_event = threading.Event()
        self.thinking_sound_active = threading.Event()
        self.tts_active = threading.Event()
        # self._interrupted removed — mid-speech wake word detection caused ALSA contention
        self.current_state = BotStates.WARMUP
        self.last_state_change = time.time()
        
        # Audio State
        self.current_audio_process = None
        self.tts_queue = []
        self.current_display_image = None  # Set when a photo/image is shown on screen

        # Memory
        self.brain = Brain()

        # Init UI
        self.background_label = tk.Label(master, bg='black')
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)

        # Mic gain meter (extracted to core/meter.py)
        self.meter = MicMeter(master)

        # BMO-themed captions: dark green text on translucent lime-green background
        self.status_label = tk.Label(
            master,
            text="Initializing...",
            font=('Courier New', 14, 'bold'),
            fg='#1a5c2a',       # Dark forest green text
            bg='#bdffcb',       # BMO's signature green
            padx=12, pady=4,
            relief='flat',
            highlightthickness=0
        )
        self.status_label.place(relx=0.5, rely=0.92, anchor=tk.S)

        # Thought bubble overlay for transcribed user input
        self.bubble = ThoughtBubble(master)

        self.is_muted = False
        self.mute_label = tk.Label(
            master,
            text="🔇 Muted",
            font=('Courier New', 16, 'bold'),
            fg='#f44336',
            bg='#bdffcb',       # BMO's signature green
            padx=10, pady=5,
            relief='flat',
            highlightthickness=0
        )

        self.animations = {}
        self.current_frame = 0
        self.load_animations()
        self.load_sounds()
        self.update_animation()

        # Start Main Thread
        threading.Thread(target=self.main_loop, daemon=True).start()
        
        # Start Screensaver Audio Thread
        self.last_screensaver_audio_time = time.time()
        threading.Thread(target=self.screensaver_audio_loop, daemon=True).start()

    def show_user_prompt(self, text):
        """Show transcribed text as an animated thought bubble, auto-hide after 8s."""
        self.bubble.show(text)

    def exit_fullscreen(self, event=None):
        self.stop_event.set()
        self.master.quit()

    def set_state(self, state, msg=""):
        if state != self.current_state:
            self.current_state = state
            self.current_frame = 0
            self.last_state_change = time.time()
            bmo_print("STATE", f"{state.upper()}: {msg}")
            # Mic gain meter: show only during LISTENING
            if state == BotStates.LISTENING:
                self.meter.show()
            else:
                self.meter.hide()
        if msg:
            self.status_label.config(text=msg)

    def mute_bmo(self, event=None):
        """Toggle audio mute and sets the whimsical 'shhh' face."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_label.place(relx=0.95, rely=0.05, anchor=tk.NE)
            # Stop persistent TTS player immediately
            player = get_player()
            if player:
                player.stop_playback()
                bmo_print("MUTE", "Stopped persistent audio player.")
            try:
                # Kill any sound effects playing via aplay
                subprocess.run(["killall", "-9", "aplay"], capture_output=True)
            except Exception:
                pass
                
            old_state = self.current_state
            self.set_state(BotStates.SHHH, "Muted")
            
            # Stop background thinking audio loops too if they're active
            if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                try:
                    self.thinking_audio_process.terminate()
                except Exception:
                    pass
                self.thinking_audio_process = None

            # After 3 seconds, resume natural state
            def revert_state():
                if self.current_state == BotStates.SHHH:
                    self.set_state(old_state if old_state != BotStates.SHHH else BotStates.IDLE, "Muted")
            self.master.after(3000, revert_state)
        else:
            self.mute_label.place_forget()
            self.set_state(BotStates.HAPPY, "Unmuted!")
            def revert_state():
                if self.current_state == BotStates.HAPPY:
                    self.set_state(BotStates.IDLE, "Ready...")
            self.master.after(2000, revert_state)

    # --- ANIMATION & SOUND ENGINE ---
    def load_sounds(self):
        self.sounds = {
            "boot_sounds": [],
            "greeting_sounds": [],
            "ack_sounds": [],
            "thinking_sounds": [],
            "analyzing_sounds": [],
            "camera_sounds": [],
            "music": []
        }
        base = "sounds"
        lang_suffix = f"_{LANGUAGE}" if LANGUAGE != "en" else ""
        for category in self.sounds.keys():
            # Try language-specific directory first (e.g. greeting_sounds_de), fall back to default
            path = os.path.join(base, category + lang_suffix)
            if not os.path.exists(path):
                path = os.path.join(base, category)
            if os.path.exists(path):
                self.sounds[category] = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.wav')]

    def play_sound(self, category):
        if self.is_muted:
            return None
        sounds = self.sounds.get(category, [])
        if not sounds:
            return None
        sound_file = random.choice(sounds)
        bmo_print("AUDIO", f"Playing {category}: {os.path.basename(sound_file)}")
        try:
            return subprocess.Popen(['aplay', '-D', ALSA_DEVICE, '-q', sound_file])
        except Exception as e:
            bmo_print("AUDIO", f"Error playing sound {sound_file}: {e}")
            return None

    def load_animations(self):
        base = "faces"
        all_face_paths = []
        for state in [BotStates.WARMUP, BotStates.IDLE, BotStates.LISTENING, BotStates.THINKING, BotStates.SPEAKING, BotStates.ERROR, BotStates.HAPPY, BotStates.SAD, BotStates.ANGRY, BotStates.SURPRISED, BotStates.SLEEPY, BotStates.DIZZY, BotStates.CHEEKY, BotStates.HEART, BotStates.STARRY_EYED, BotStates.CONFUSED, BotStates.SHHH, BotStates.JAMMING, BotStates.FOOTBALL, BotStates.DETECTIVE, BotStates.SIR_MANO, BotStates.LOW_BATTERY, BotStates.BEE, BotStates.CAPTURING]:
            path = os.path.join(base, state)
            self.animations[state] = []
            if os.path.exists(path):
                files = sorted([f for f in os.listdir(path) if f.lower().endswith('.png')])
                for f in files:
                    img_path = os.path.join(path, f)
                    img = Image.open(img_path).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
                    
        # Load screensaver as full animation sequences per expression
        # Only include expressions that make sense without audio context
        SCREENSAVER_STATES = [
            "idle", "happy", "sleepy", "heart", "starry_eyed",
            "cheeky", "dizzy", "confused",
            "daydream", "bored", "jamming", "curious",
            "football", "detective", "sir_mano", "low_battery", "bee"
        ]
        self.screensaver_sequences = []  # List of (state_name, [frames])
        for state_dir in SCREENSAVER_STATES:
            path = os.path.join(base, state_dir)
            if not os.path.isdir(path):
                continue
            files = sorted([f for f in os.listdir(path) if f.lower().endswith('.png')])
            if files:
                seq_frames = []
                for f in files:
                    try:
                        img = Image.open(os.path.join(path, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                        seq_frames.append(ImageTk.PhotoImage(img))
                    except Exception as e:
                        bmo_print("SCREENSAVER", f"Failed to load image {f}: {e}")
                if seq_frames:
                    self.screensaver_sequences.append((state_dir, seq_frames))
        
        # Build the screensaver animation: play each expression's full sequence
        random.shuffle(self.screensaver_sequences)
        self.animations[BotStates.SCREENSAVER] = []
        for name, seq in self.screensaver_sequences:
            # Play each expression's sequence 2x so you can see the animation
            self.animations[BotStates.SCREENSAVER].extend(seq * 2)
    
    def update_animation(self):
        if self.current_state == BotStates.DISPLAY_IMAGE or self.current_display_image is not None:
            # Don't animate — a photo or image is being shown
            self.master.after(500, self.update_animation)
            return

        # Check for screensaver trigger
        if self.current_state == BotStates.IDLE and (time.time() - self.last_state_change) > 60:
            self.set_state(BotStates.SCREENSAVER, "Screensaver...")

        # If entering listening from screensaver, immediately break out
        if self.current_state == BotStates.LISTENING and self.current_frame > 0 and 'screensaver' in str(self.animations.get(self.current_state, [])):
            self.current_frame = 0 # reset cleanly

        # Hide text labels during screensaver
        if self.current_state == BotStates.SCREENSAVER:
            if self.status_label.winfo_ismapped():
                self.status_label.place_forget()
            self.bubble.hide()
        else:
            if not self.status_label.winfo_ismapped():
                self.status_label.place(relx=0.5, rely=0.92, anchor=tk.S)

        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if frames:
            if self.current_state == BotStates.WARMUP:
                # Cycle symbol eye frames (0-11) at 1s each, don't advance past frame 12
                # (frames 12+ are controlled by the boot sequence when LLM finishes)
                if self.current_frame < 12:
                    self.current_frame = (self.current_frame + 1) % 12
            else:
                self.current_frame = (self.current_frame + 1) % len(frames)
            
            # Re-shuffle screensaver sequences when loop completes
            if self.current_state == BotStates.SCREENSAVER and self.current_frame == 0:
                random.shuffle(self.screensaver_sequences)
                self.animations[BotStates.SCREENSAVER] = []
                for name, seq in self.screensaver_sequences:
                    self.animations[BotStates.SCREENSAVER].extend(seq * 2)
                
            self.background_label.config(image=frames[min(self.current_frame, len(frames) - 1)])
        
        # Match web UI animation speeds
        speed = 500
        if self.current_state == BotStates.WARMUP:
            speed = 1000  # One symbol rotation per second
        elif self.current_state == BotStates.SPEAKING:
            speed = 90   # Fix: 90ms for natural lip sync
        elif self.current_state == BotStates.THINKING:
            speed = 500
        elif self.current_state == BotStates.LISTENING:
            speed = 250
        elif self.current_state == BotStates.CAPTURING:
            speed = 150  # 8 frames × 150ms = 1.2s per shutter cycle
        elif self.current_state == BotStates.SCREENSAVER or self.current_state == BotStates.SHHH:
            speed = 400 # Smooth animation speed for sequences

        self.master.after(speed, self.update_animation)

    # --- AUDIO INPUT ---
    def wait_for_wakeword(self, oww):
        """Block until wake word is heard."""
        suppressed = {BotStates.SPEAKING, BotStates.JAMMING, BotStates.LISTENING}
        result = wait_for_wakeword(oww, self.stop_event, lambda: self.current_state, suppressed)
        if not result and not self.stop_event.is_set():
            self.set_state(BotStates.ERROR)
            time.sleep(2)
        return result

    def record_audio(self):
        """Record until silence"""
        bmo_print("STT", "Recording...")
        return record_until_silence(
            self.stop_event,
            meter_cb=self.meter.feed,
            grace_sec=1.5,
            timeout_sec=30.0,
            filename="input.wav",
        )

    # --- TIMERS & REMINDERS ---
    def start_timer_thread(self, minutes, message):
        def timer_worker():
            bmo_print("TIMER SET", f"for {minutes} minutes. Message: {message}")
            time.sleep(minutes * 60)
            bmo_print("TIMER DONE", message)
            
            # Wait for BMO to finish speaking/listening to avoid ALSA conflicts
            while self.current_state in [BotStates.SPEAKING, BotStates.LISTENING]:
                time.sleep(1)
                
            # Interject the alarm
            old_state = self.current_state
            self.set_state(BotStates.HAPPY, "Reminder!")
            # Play an alert noise if we have one
            alert_proc = self.play_sound("ack_sounds")
            if alert_proc:
                alert_proc.wait()
                
            self.speak(message, msg="Reminder!")
            
            # Return BMO to whatever they were doing (e.g. IDLE or SCREENSAVER)
            time.sleep(1)
            if self.current_state == BotStates.IDLE:
                self.set_state(old_state if old_state != BotStates.HAPPY else BotStates.IDLE, "Ready")
                
        threading.Thread(target=timer_worker, daemon=True).start()

    # --- STT & TTS ---
    def transcribe(self, filename):
        bmo_print("STT", "Transcribing...")
        return transcribe_audio(filename)

    def speak(self, text, msg="Speaking..."):
        clean_text = clean_text_for_speech(text)
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return

        bmo_print("TTS", f"Speaking: {clean_text[:30]}...")
        try:
            synth = get_synthesizer()
            player = get_player()

            # 1. Synthesize audio first. (Runs silently, mouth stays idle/thinking)
            pcm = synth.synthesize(clean_text)
            if not pcm:
                bmo_print("TTS", "Piper returned no audio")
                return

            # 2. Audio is ready! Set SPEAKING state so mouth starts moving.
            if msg is not None:
                self.set_state(BotStates.SPEAKING, msg)
            else:
                self.current_state = BotStates.SPEAKING
                self.current_frame = 0
                self.last_state_change = time.time()

            # 3. Play through persistent stream
            if not self.is_muted:
                player.resume()
                player.play(pcm)
            else:
                # If muted, just hold the speaking pose for a moment to simulate talking
                time.sleep(1.5)

            # 4. Enforce an immediate IDLE state and a short visual breath pause
            # This ensures the mouth closes definitively before the next sentence chunk arrives,
            # unless a background thread (like play_music) has already hijacked the state to JAMMING
            if self.current_state == BotStates.SPEAKING:
                if msg is not None:
                    self.set_state(BotStates.IDLE, "Ready...")
                else:
                    self.current_state = BotStates.IDLE
                    self.current_frame = 0
                    self.last_state_change = time.time()
                time.sleep(0.3)

        except Exception as e:
            bmo_print("TTS", f"Hardware error: {e}")

    def record_followup(self, timeout_sec=8):
        """Listen briefly for a follow-up question after BMO responds."""
        bmo_print("FOLLOW-UP", "Listening...")
        return record_until_silence(
            self.stop_event,
            meter_cb=self.meter.feed,
            grace_sec=timeout_sec,
            timeout_sec=timeout_sec + 8,
            ignore_sec=1.0,
            filename="followup.wav",
        )

    def apply_bmo_border(self, pil_img):
        """Resize, crop, and add BMO-style border to an image for LCD display."""
        from PIL import ImageOps
        lcd_w, lcd_h = self.BG_WIDTH - 60, self.BG_HEIGHT - 60
        img_ratio = pil_img.width / pil_img.height
        target_ratio = lcd_w / lcd_h
        if img_ratio > target_ratio:
            new_h = lcd_h
            new_w = int(new_h * img_ratio)
        else:
            new_w = lcd_w
            new_h = int(new_w / img_ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - lcd_w) / 2
        top = (new_h - lcd_h) / 2
        right = (new_w + lcd_w) / 2
        bottom = (new_h + lcd_h) / 2
        pil_img = pil_img.crop((left, top, right, bottom))
        pil_img = ImageOps.expand(pil_img, border=10, fill="#1c201a")
        pil_img = ImageOps.expand(pil_img, border=20, fill="#38b5a0")
        return pil_img


    # --- MAIN LOOP ---
    def main_loop(self):
        time.sleep(1) # Let UI settle
        boot_start = time.time()

        # Initialize persistent audio subsystem (TTS player + Piper synthesizer)
        self.set_state(BotStates.WARMUP, "Booting...")
        t0 = time.time()
        init_audio(ALSA_DEVICE)
        atexit.register(shutdown_audio)
        bmo_print("BOOT", f"Audio subsystem: {time.time()-t0:.2f}s")

        # Play a boot sound so the user knows BMO is starting up (~15s total)
        self.play_sound("boot_sounds")
        # Frame 1-2: empty face → brain bar appears
        self.current_frame = 1

        # Initialize LLM on the Hailo NPU — this is the big one (~12s)
        # Advance warmup frames during loading to show progress
        self.set_state(BotStates.WARMUP, "Loading Brain...")
        t0 = time.time()

        def _advance_warmup():
            """Let frames auto-cycle during LLM load — no manual advance needed."""
            pass  # Animation loop handles 0-11 cycling at 1s each
        warmup_thread = threading.Thread(target=_advance_warmup, daemon=True)
        warmup_thread.start()

        init_llm()
        bmo_print("BOOT", f"LLM (NPU): {time.time()-t0:.2f}s")

        # Frame 13: eyes closed — loading ears
        self.current_frame = 12
        self.set_state(BotStates.WARMUP, "Loading Ears...")
        t0 = time.time()
        init_stt()
        bmo_print("BOOT", f"STT (NPU): {time.time()-t0:.2f}s")

        # Frame 14: eyes half open + mouth — loading wake word
        self.current_frame = 13
        self.set_state(BotStates.WARMUP, "Loading Wake Word...")
        t0 = time.time()
        try:
            oww = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
        except Exception as e:
            bmo_print("WAKE", f"Failed to load model: {e}")
            self.set_state(BotStates.ERROR, "Wake Word Error")
            return
        bmo_print("BOOT", f"Wake word (OWW): {time.time()-t0:.2f}s")

        # Frame 15-16: fully assembled → happy face
        self.current_frame = 14
        time.sleep(0.5)
        self.current_frame = 15
        time.sleep(0.5)

        bmo_print("BOOT", f"Total startup: {time.time()-boot_start:.2f}s")
        self.set_state(BotStates.SPEAKING, "Ready!")
        greeting_proc = self.play_sound("greeting_sounds")
        if greeting_proc:
            # Wait for greeting to finish before going idle
            threading.Thread(target=lambda: (greeting_proc.wait(), self.set_state(BotStates.IDLE, "Waiting...") if self.current_state == BotStates.SPEAKING else None), daemon=True).start()
        else:
            self.set_state(BotStates.IDLE, "Waiting...")

        while not self.stop_event.is_set():
            # 1. Wait for Wake Word
            if self.wait_for_wakeword(oww):
                # 2. Record
                self.set_state(BotStates.LISTENING, "Listening...")
                wav_file = self.record_audio()
                
                # 3. Transcribe
                self.set_state(BotStates.THINKING, "Transcribing...")
                
                def play_thinking_sequence():
                    ack_proc = self.play_sound("ack_sounds")
                    if ack_proc:
                        ack_proc.wait()
                    
                    while self.current_state == BotStates.THINKING:
                        self.thinking_audio_process = self.play_sound("thinking_sounds")
                        if self.thinking_audio_process:
                            self.thinking_audio_process.wait()
                        # Wait 8 seconds before playing again, but check state frequently
                        for _ in range(80):
                            if self.current_state != BotStates.THINKING:
                                break
                            time.sleep(0.1)
                
                threading.Thread(target=play_thinking_sequence, daemon=True).start()

                user_text = self.transcribe(wav_file)
                bmo_print("STT", f"User Transcribed: {user_text}")
                self.show_user_prompt(user_text)

                if len(user_text) < 2:
                    self.set_state(BotStates.IDLE, "Ready")
                    if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                        try:
                            self.thinking_audio_process.terminate()
                        except Exception:
                            pass
                        self.thinking_audio_process = None
                    continue

                # 4. LLM
                self.set_state(BotStates.THINKING, "Thinking...")

                # Stop the thinking sound loop
                if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                    try:
                        self.thinking_audio_process.terminate()
                    except Exception:
                        pass
                    self.thinking_audio_process = None

                try:
                    valid_exprs = {BotStates.HAPPY, BotStates.SAD, BotStates.ANGRY,
                                   BotStates.SURPRISED, BotStates.SLEEPY, BotStates.DIZZY,
                                   BotStates.CHEEKY, BotStates.HEART, BotStates.STARRY_EYED,
                                   BotStates.CONFUSED}

                    def on_music():
                        def music_worker():
                            while self.current_state in [BotStates.SPEAKING, BotStates.THINKING]:
                                time.sleep(0.5)
                            self.speak(random.choice(t("music_intros")), msg="Getting ready to jam...")
                            bmo_print("MUSIC", "Starting music playback...")
                            music_proc = self.play_sound("music")
                            if music_proc:
                                self.set_state(BotStates.JAMMING, "Jamming!")
                                bmo_print("MUSIC", "Now playing! State set to JAMMING")
                                music_proc.wait()
                                bmo_print("MUSIC", "Playback finished")
                                time.sleep(1)
                                if self.current_state == BotStates.JAMMING:
                                    self.set_state(BotStates.IDLE, "Ready")
                            else:
                                bmo_print("MUSIC", "No music files found or muted!")
                                self.speak(t("no_music"))
                        threading.Thread(target=music_worker, daemon=True).start()

                    result = dispatch_stream(
                        self.brain, user_text,
                        on_expression=lambda expr: self.set_state(expr, f"Feeling {expr}..."),
                        on_timer=self.start_timer_thread,
                        on_music=on_music,
                        valid_expressions=valid_exprs,
                    )

                    for chunk in result.speak_chunks:
                        self.speak(chunk)

                    if result.take_photo:
                        # Stop thinking sound loop by changing state (breaks the while loop)
                        # then kill any currently playing sound
                        self.set_state(BotStates.CAPTURING, "Camera mode!")
                        if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                            try:
                                self.thinking_audio_process.terminate()
                            except Exception:
                                pass
                            self.thinking_audio_process = None
                        time.sleep(0.2)  # let aplay process exit
                        # Clear transcription bubble so it doesn't clip the camera face
                        self.bubble.hide()
                        # Start releasing LLM+STT in background while camera UX plays
                        # (speech + aplay don't use the NPU, so this is safe)
                        release_thread = threading.Thread(target=_release_llm, daemon=True)
                        release_thread.start()
                        # --- Camera UX: spoken intro + shutter ---
                        self.speak(random.choice(t("camera_intros")))
                        self.set_state(BotStates.CAPTURING, "Say cheese!")
                        time.sleep(2)
                        self.play_sound("camera_sounds")
                        try:
                            # Try libcamera-still (older) or rpicam-still (newer Pi OS)
                            cam_cmd = None
                            import shutil
                            for candidate in ['libcamera-still', 'rpicam-still']:
                                if shutil.which(candidate):
                                    cam_cmd = candidate
                                    break
                            if cam_cmd is None:
                                raise FileNotFoundError("No camera command found (libcamera-still / rpicam-still)")
                            subprocess.run([cam_cmd, '-o', 'temp.jpg', '--width', '640', '--height', '480', '--nopreview', '-t', '1000'], check=True)
                            import base64
                            with open('temp.jpg', 'rb') as img_file:
                                b64_string = base64.b64encode(img_file.read()).decode('utf-8')

                            # Display captured photo immediately
                            try:
                                photo_img = Image.open('temp.jpg')
                                photo_img = self.apply_bmo_border(photo_img)
                                self.current_display_image = ImageTk.PhotoImage(photo_img)
                                self.background_label.config(image=self.current_display_image)
                            except Exception as e:
                                bmo_print("CAMERA", f"Photo display error: {e}")

                            self.set_state(BotStates.THINKING, "Analyzing...")

                            def play_analyzing_sequence():
                                # Loop analyzing sounds only — no thinking sounds during image analysis
                                while self.current_state == BotStates.THINKING:
                                    self.thinking_audio_process = self.play_sound("analyzing_sounds")
                                    if self.thinking_audio_process:
                                        self.thinking_audio_process.wait()
                                    # Pause between clips
                                    for _ in range(50):
                                        if self.current_state != BotStates.THINKING:
                                            break
                                        time.sleep(0.1)

                            threading.Thread(target=play_analyzing_sequence, daemon=True).start()
                            # Ensure NPU is fully released before VLM subprocess claims it
                            release_thread.join(timeout=10)
                            response = self.brain.analyze_image(b64_string, user_text)
                            if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                                try:
                                    self.thinking_audio_process.terminate()
                                except Exception:
                                    pass
                                self.thinking_audio_process = None
                            # Photo is already displayed — speak VLM response over it
                            self.speak(response)

                            # Keep photo displayed during LLM+STT reload
                            self.set_state(BotStates.WARMUP, "Reloading Brain...")
                            self.play_sound("boot_sounds")
                            reload_after_vlm()

                            # Restore face animation after reload complete
                            self.current_display_image = None
                        except FileNotFoundError as e:
                            bmo_print("CAMERA", f"Error: {e}")
                            self.speak(t("no_camera"))

                        except Exception as e:
                            bmo_print("CAMERA", f"Error: {e}")
                            self.speak(t("camera_error"))
                    
                    # 5. Display Image (if any)
                    if result.image_url:
                        image_url = result.image_url
                        # Speak confirmation before downloading
                        self.speak(t("draw_image"))
                        self.set_state(BotStates.DISPLAY_IMAGE, "Showing Image...")
                        bmo_print("IMAGE", f"Starting image display for: {image_url}")
                        try:
                            # Migrate broken gen.pollinations.ai URL to working image.pollinations.ai
                            image_url = image_url.replace("gen.pollinations.ai/image/", "image.pollinations.ai/prompt/")
                            bmo_print("IMAGE", f"Downloading: {image_url}")
                            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=15) as u:
                                raw_data = u.read()
                            bmo_print("IMAGE", f"Downloaded: {len(raw_data)} bytes")
                            from io import BytesIO

                            img = Image.open(BytesIO(raw_data))
                            img = self.apply_bmo_border(img)
                            
                            # Schedule Tkinter update on main thread for thread safety
                            def show_image(pil_img=img):
                                try:
                                    self.current_display_image = ImageTk.PhotoImage(pil_img)
                                    self.background_label.config(image=self.current_display_image)
                                    bmo_print("IMAGE", "Displayed on screen")
                                except Exception as e:
                                    bmo_print("IMAGE", f"Tkinter display error: {e}")
                            
                            self.master.after(0, show_image)
                        except Exception as e:
                            bmo_print("IMAGE", f"Download/Display Error: {e}")

                except Exception as e:
                    bmo_print("ERROR", f"LLM/TTS pipeline: {e}")
                    traceback.print_exc()

                self.set_state(BotStates.IDLE, "Ready")

                # Skip follow-up listening when music was triggered — the mic would
                # pick up BMO's own intro speech and the music itself as false input.
                if result.music_triggered:
                    bmo_print("FOLLOW-UP", "Skipped — music playback active")
                    continue

                if not FOLLOWUP_ENABLED:
                    bmo_print("FOLLOW-UP", "Disabled via config")
                    continue

                # Conversation follow-up: let user reply repeatedly as long as they respond within 8 seconds
                while True:
                    self.set_state(BotStates.LISTENING, "Still listening...")
                    followup_wav = self.record_followup(timeout_sec=8)
                    
                    if not followup_wav:
                        # User didn't reply within 8 seconds, end conversation thread
                        self.set_state(BotStates.IDLE, "Waiting...")
                        break
                        
                    self.set_state(BotStates.THINKING, "Transcribing...")
                    threading.Thread(target=play_thinking_sequence, daemon=True).start()
                    user_text = self.transcribe(followup_wav)
                    bmo_print("STT", f"Follow-up Transcribed: {user_text}")
                    self.show_user_prompt(user_text)

                    if len(user_text) < 2:
                        # Mic picked up noise, but no actual speech. End conversation.
                        if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                            try:
                                self.thinking_audio_process.terminate()
                            except Exception:
                                pass
                            self.thinking_audio_process = None
                        self.set_state(BotStates.IDLE, "Waiting...")
                        break

                    self.set_state(BotStates.THINKING, "Thinking...")
                    if hasattr(self, 'thinking_audio_process') and self.thinking_audio_process:
                        try:
                            self.thinking_audio_process.terminate()
                        except Exception:
                            pass
                        self.thinking_audio_process = None

                    try:
                        followup_result = dispatch_stream(
                            self.brain, user_text,
                            on_expression=lambda expr: self.set_state(expr, f"Feeling {expr}..."),
                            on_timer=self.start_timer_thread,
                            on_music=on_music,
                            valid_expressions=valid_exprs,
                        )
                        for chunk in followup_result.speak_chunks:
                            self.speak(chunk)
                    except Exception as e:
                        bmo_print("AGENT", f"Follow-up LLM error: {e}")

                    self.set_state(BotStates.IDLE, "Ready")
                    # Loop back around and listen again!
                    
                # Guarantee a 1 second cool-down before we loop all the way back up
                # and call wait_for_wakeword(). This ensures ALSA capture locks are fully
                # released by the kernel, preventing PaErrorCode -9999 crashes.
                time.sleep(1.0)

    def screensaver_audio_loop(self):
        def display_image_cb(img_url):
            """Download and display an image on the BMO screen."""
            bmo_print("SCREENSAVER", f"Downloading image from: {img_url}")
            self.set_state(BotStates.DISPLAY_IMAGE, "Visualizing...")
            try:
                req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as u:
                    raw_data = u.read()
                bmo_print("SCREENSAVER", f"Image downloaded: {len(raw_data)} bytes")
                from io import BytesIO
                from PIL import ImageOps
                img = Image.open(BytesIO(raw_data))
                img = self.apply_bmo_border(img)
                def show(pil_img=img):
                    try:
                        self.current_display_image = ImageTk.PhotoImage(pil_img)
                        self.background_label.config(image=self.current_display_image)
                    except Exception as e:
                        bmo_print("SCREENSAVER", f"Tkinter display error: {e}")
                self.master.after(0, show)
                time.sleep(10)
                if self.current_state == BotStates.DISPLAY_IMAGE:
                    self.set_state(BotStates.SCREENSAVER, "Sleeping...")
            except Exception as e:
                bmo_print("SCREENSAVER", f"Image display error: {e}")
                if self.current_state == BotStates.DISPLAY_IMAGE:
                    self.set_state(BotStates.SCREENSAVER, "Sleeping...")

        screensaver_loop(
            stop_event=self.stop_event,
            get_state=lambda: self.current_state,
            set_state=self.set_state,
            speak_fn=self.speak,
            play_sound_fn=self.play_sound,
            is_muted_fn=lambda: self.is_muted,
            display_image_fn=display_image_cb,
            get_last_state_change=lambda: self.last_state_change,
            get_last_audio_time=lambda: self.last_screensaver_audio_time,
            set_last_audio_time=lambda t: setattr(self, 'last_screensaver_audio_time', t),
            is_llm_ready_fn=is_llm_ready,
            get_llm_fn=_get_llm,
            screensaver_state=BotStates.SCREENSAVER,
            display_image_state=BotStates.DISPLAY_IMAGE,
            expression_states=[BotStates.HEART, BotStates.SLEEPY, BotStates.STARRY_EYED, BotStates.DIZZY],
            persona_states=[BotStates.FOOTBALL, BotStates.DETECTIVE, BotStates.SIR_MANO, BotStates.LOW_BATTERY, BotStates.BEE],
            alsa_device=ALSA_DEVICE,
        )

if __name__ == "__main__":
    import signal
    root = tk.Tk()
    app = BotGUI(root)
    signal.signal(signal.SIGINT, lambda *_: (app.stop_event.set(), root.quit()))
    root.mainloop()
