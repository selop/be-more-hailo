"""Image download, border-framing, and Tkinter display for BMO."""

import urllib.request
from io import BytesIO

from PIL import Image, ImageOps, ImageTk

from .log import bmo_print


def apply_bmo_border(pil_img, width=800, height=480):
    """Resize, crop, and add BMO-style border to an image for LCD display."""
    lcd_w, lcd_h = width - 60, height - 60
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
    pil_img = pil_img.crop((left, top, left + lcd_w, top + lcd_h))
    pil_img = ImageOps.expand(pil_img, border=10, fill="#1c201a")
    pil_img = ImageOps.expand(pil_img, border=20, fill="#38b5a0")
    return pil_img


def download_and_display(image_url, background_label, master, set_display_image):
    """Download an image URL, apply BMO border, and show it on the Tkinter label.

    Parameters
    ----------
    image_url : str
        The URL to fetch.
    background_label : tk.Label
        The label to display the image on.
    master : tk.Tk
        The Tkinter root (for thread-safe ``after()`` scheduling).
    set_display_image : callable(ImageTk.PhotoImage)
        Callback to store the PhotoImage reference (prevents GC).

    Returns True on success, False on failure.
    """
    try:
        # Migrate broken gen.pollinations.ai URL to working image.pollinations.ai
        image_url = image_url.replace(
            "gen.pollinations.ai/image/", "image.pollinations.ai/prompt/"
        )
        bmo_print("IMAGE", f"Downloading: {image_url}")
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as u:
            raw_data = u.read()
        bmo_print("IMAGE", f"Downloaded: {len(raw_data)} bytes")

        img = Image.open(BytesIO(raw_data))
        img = apply_bmo_border(img)

        def show(pil_img=img):
            try:
                tk_img = ImageTk.PhotoImage(pil_img)
                set_display_image(tk_img)
                background_label.config(image=tk_img)
                bmo_print("IMAGE", "Displayed on screen")
            except Exception as e:
                bmo_print("IMAGE", f"Tkinter display error: {e}")

        master.after(0, show)
        return True
    except Exception as e:
        bmo_print("IMAGE", f"Download/Display Error: {e}")
        return False
