"""Camera capture, VLM analysis, and photo display for BMO."""

import base64
import random
import shutil
import subprocess
import threading
import time

from PIL import Image, ImageTk

from .config import t
from .image_display import apply_bmo_border
from .npu import _release_llm, reload_after_vlm
from .log import bmo_print


class CameraHandler:
    """Encapsulates the full camera-capture → VLM-analysis → reload cycle.

    Parameters
    ----------
    set_state : callable(str, str)
    speak : callable(str, msg=str)
    play_sound : callable(str) -> Popen | None
    brain : Brain
    background_label : tk.Label
    bubble : ThoughtBubble
    set_display_image : callable(ImageTk.PhotoImage)
        Store the PhotoImage reference to prevent GC.
    clear_display_image : callable()
        Set the display image reference to None.
    stop_thinking_audio : callable()
    """

    def __init__(
        self,
        *,
        set_state,
        speak,
        play_sound,
        brain,
        background_label,
        bubble,
        set_display_image,
        clear_display_image,
        stop_thinking_audio,
    ):
        self.set_state = set_state
        self.speak = speak
        self.play_sound = play_sound
        self.brain = brain
        self.label = background_label
        self.bubble = bubble
        self.set_display_image = set_display_image
        self.clear_display_image = clear_display_image
        self.stop_thinking_audio = stop_thinking_audio

    def handle_photo(self, user_text, get_state):
        """Run the full camera UX: intro speech → capture → VLM → reload.

        Should be called from the main conversation thread when
        ``ActionResult.take_photo`` is True.
        """
        self.set_state("capturing", "Camera mode!")
        self.stop_thinking_audio()
        time.sleep(0.2)  # let aplay process exit

        # Clear transcription bubble so it doesn't clip the camera face
        self.bubble.hide()

        # Start releasing LLM+STT in background while camera UX plays
        release_thread = threading.Thread(target=_release_llm, daemon=True)
        release_thread.start()

        # --- Camera UX: spoken intro + shutter ---
        self.speak(random.choice(t("camera_intros")))
        self.set_state("capturing", "Say cheese!")
        time.sleep(2)
        self.play_sound("camera_sounds")

        try:
            cam_cmd = None
            for candidate in ["libcamera-still", "rpicam-still"]:
                if shutil.which(candidate):
                    cam_cmd = candidate
                    break
            if cam_cmd is None:
                raise FileNotFoundError(
                    "No camera command found (libcamera-still / rpicam-still)"
                )

            subprocess.run(
                [cam_cmd, "-o", "temp.jpg", "--width", "640", "--height", "480",
                 "--nopreview", "-t", "1000"],
                check=True,
            )

            with open("temp.jpg", "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode("utf-8")

            # Display captured photo immediately
            try:
                photo_img = Image.open("temp.jpg")
                photo_img = apply_bmo_border(photo_img)
                tk_img = ImageTk.PhotoImage(photo_img)
                self.set_display_image(tk_img)
                self.label.config(image=tk_img)
            except Exception as e:
                bmo_print("CAMERA", f"Photo display error: {e}")

            self.set_state("thinking", "Analyzing...")

            # Analyzing sound loop
            analyzing_stop = threading.Event()

            def play_analyzing_sequence():
                while not analyzing_stop.is_set() and get_state() == "thinking":
                    proc = self.play_sound("analyzing_sounds")
                    if proc:
                        proc.wait()
                    for _ in range(50):
                        if analyzing_stop.is_set() or get_state() != "thinking":
                            break
                        time.sleep(0.1)

            threading.Thread(target=play_analyzing_sequence, daemon=True).start()

            # Ensure NPU is fully released before VLM subprocess claims it
            release_thread.join(timeout=10)
            response = self.brain.analyze_image(b64_string, user_text)

            analyzing_stop.set()

            # Speak VLM response over the displayed photo
            self.speak(response)

            # Reload LLM+STT while photo stays on screen
            self.set_state("warmup", "Reloading Brain...")
            self.play_sound("boot_sounds")
            reload_after_vlm()

            # Restore face animation
            self.clear_display_image()

        except FileNotFoundError as e:
            bmo_print("CAMERA", f"Error: {e}")
            self.speak(t("no_camera"))

        except Exception as e:
            bmo_print("CAMERA", f"Error: {e}")
            self.speak(t("camera_error"))
