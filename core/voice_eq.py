# core/voice_eq.py — Touch-screen Voice EQ config overlay
"""Modal overlay for tuning BMO's voice low-pass filter in real-time.

Full-screen CRT-green aesthetic matching BMO's Adventure Time look.
Two sliders (cutoff + order) with draggable cutoff dot on the curve.
"""

import json
import logging
import math
import os
import threading
import tkinter as tk

from . import config

logger = logging.getLogger(__name__)

# ── Persistence helpers ──────────────────────────────────────────────────

def voice_eq_load() -> dict:
    """Read voice_eq.json (or return current config as defaults)."""
    if os.path.exists(config.VOICE_EQ_FILE):
        try:
            with open(config.VOICE_EQ_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "enabled": config.VOICE_LPF_ENABLED,
        "cutoff": config.VOICE_LPF_CUTOFF,
        "order": config.VOICE_LPF_ORDER,
    }


def voice_eq_save(enabled: bool, cutoff: int, order: int):
    """Write voice_eq.json and update running config globals + filter cache."""
    data = {"enabled": enabled, "cutoff": cutoff, "order": order}
    try:
        with open(config.VOICE_EQ_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"voice_eq_save: {e}")
    config.VOICE_LPF_ENABLED = enabled
    config.VOICE_LPF_CUTOFF = cutoff
    config.VOICE_LPF_ORDER = order
    from .tts import PiperSynthesizer, invalidate_cache
    PiperSynthesizer.invalidate_lpf()
    invalidate_cache()  # pre-cached WAVs need re-filtering with new settings


# ── Overlay widget ───────────────────────────────────────────────────────

class VoiceEQOverlay:
    """Full-screen voice EQ overlay with CRT-green Adventure Time aesthetic."""

    SCREEN_W, SCREEN_H = 800, 480

    # CRT green palette
    BG          = '#3d6b3d'
    BORDER      = '#7dff7d'
    BORDER_W    = 8
    INNER_BG    = '#2d5a2d'
    LINE_CLR    = '#7dff7d'
    DIM_CLR     = '#4a8a4a'
    CURVE_CLR   = '#ff3333'
    DOT_CLR     = '#ff4444'
    DOT_R       = 10
    SLIDER_TRACK = '#1a3a1a'
    SLIDER_FILL  = '#7dff7d'
    SLIDER_THUMB = '#ffffff'
    BTN_BG      = '#2d5a2d'
    BTN_OUTLINE = '#7dff7d'
    BTN_TEXT    = '#7dff7d'
    BTN_OFF_BG  = '#5a2d2d'

    # Curve geometry
    CURVE_X, CURVE_Y = 60, 55
    CURVE_W = SCREEN_W - 120   # 680px
    CURVE_H = 200

    # Slider geometry
    SLIDER_X = 140
    SLIDER_W = 480
    THUMB_R  = 20              # finger-friendly
    TRACK_H  = 6

    # Ranges
    F_MIN, F_MAX = 100.0, 11000.0
    CUTOFF_MIN, CUTOFF_MAX = 2000, 10000
    CUTOFF_STEP = 100
    ORDER_MIN, ORDER_MAX = 2, 6

    def __init__(self, master):
        self._master = master
        self._canvas = None
        self._visible = False
        eq = voice_eq_load()
        self._enabled = eq["enabled"]
        self._cutoff = eq["cutoff"]
        self._order = eq["order"]
        self._dragging = None  # 'cutoff', 'order', or 'dot'

    # -- public API --------------------------------------------------------

    def is_visible(self) -> bool:
        return self._visible

    def show(self):
        if self._visible:
            return
        eq = voice_eq_load()
        self._enabled = eq["enabled"]
        self._cutoff = eq["cutoff"]
        self._order = eq["order"]
        self._visible = True
        self._build()

    def hide(self):
        self._visible = False
        if self._canvas:
            self._canvas.place_forget()
            self._canvas.destroy()
            self._canvas = None

    # -- build UI ----------------------------------------------------------

    def _build(self):
        if self._canvas:
            self._canvas.destroy()
        self._buttons = []

        c = tk.Canvas(
            self._master, width=self.SCREEN_W, height=self.SCREEN_H,
            bg=self.BG, highlightthickness=0, bd=0)
        c.place(x=0, y=0)
        self._canvas = c

        # Bright lime border
        b = self.BORDER_W
        c.create_rectangle(
            b // 2, b // 2, self.SCREEN_W - b // 2, self.SCREEN_H - b // 2,
            outline=self.BORDER, width=b)

        # Inner darker area
        m = b + 6
        c.create_rectangle(
            m, m, self.SCREEN_W - m, self.SCREEN_H - m,
            fill=self.INNER_BG, outline=self.DIM_CLR, width=1)

        # Title
        c.create_text(
            self.SCREEN_W // 2, 32, text="BMO VOICE TUNER",
            font=('Courier New', 18, 'bold'), fill=self.LINE_CLR)

        # Frequency response curve
        self._draw_curve()

        # Cutoff slider
        cutoff_y = self.CURVE_Y + self.CURVE_H + 35
        c.create_text(
            self.SLIDER_X - 10, cutoff_y, text="Cutoff:", anchor='e',
            font=('Courier New', 12, 'bold'), fill=self.LINE_CLR)
        self._cutoff_val_id = c.create_text(
            self.SLIDER_X + self.SLIDER_W + 10, cutoff_y,
            text=f"{self._cutoff} Hz", anchor='w',
            font=('Courier New', 12, 'bold'), fill=self.CURVE_CLR)
        self._cutoff_y = cutoff_y
        self._draw_slider('cutoff')

        # Order slider
        order_y = cutoff_y + 55
        c.create_text(
            self.SLIDER_X - 10, order_y, text="Order:", anchor='e',
            font=('Courier New', 12, 'bold'), fill=self.LINE_CLR)
        self._order_val_id = c.create_text(
            self.SLIDER_X + self.SLIDER_W + 10, order_y,
            text=str(self._order), anchor='w',
            font=('Courier New', 12, 'bold'), fill=self.CURVE_CLR)
        self._order_y = order_y
        self._draw_slider('order')

        # Buttons row
        btn_y = self.SCREEN_H - 58
        btn_w, btn_h = 120, 44
        gap = 24
        total = 4 * btn_w + 3 * gap
        bx = (self.SCREEN_W - total) // 2

        self._draw_button(bx, btn_y, btn_w, btn_h,
                          "ON" if self._enabled else "OFF",
                          self._toggle_enabled,
                          bg=self.BTN_BG if self._enabled else self.BTN_OFF_BG)
        bx += btn_w + gap
        self._draw_button(bx, btn_y, btn_w, btn_h, "PREVIEW", self._preview)
        bx += btn_w + gap
        self._draw_button(bx, btn_y, btn_w, btn_h, "SAVE", self._save)
        bx += btn_w + gap
        self._draw_button(bx, btn_y, btn_w, btn_h, "CANCEL", self._cancel)

        # Canvas events — return "break" to prevent propagation to master
        c.bind('<ButtonPress-1>', self._canvas_press)
        c.bind('<B1-Motion>', self._canvas_drag)
        c.bind('<ButtonRelease-1>', self._canvas_release)

    # -- frequency response curve ------------------------------------------

    def _freq_to_x(self, freq):
        return self.CURVE_X + self.CURVE_W * (
            math.log10(freq / self.F_MIN) / math.log10(self.F_MAX / self.F_MIN))

    def _x_to_freq(self, x):
        frac = (x - self.CURVE_X) / self.CURVE_W
        frac = max(0.0, min(1.0, frac))
        return self.F_MIN * (self.F_MAX / self.F_MIN) ** frac

    def _db_to_y(self, db):
        db = max(db, -30.0)
        return self.CURVE_Y + self.CURVE_H * (-db / 30.0)

    def _draw_curve(self):
        c = self._canvas
        cx, cy, cw, ch = self.CURVE_X, self.CURVE_Y, self.CURVE_W, self.CURVE_H
        c.delete('curve')

        # Background
        c.create_rectangle(
            cx, cy, cx + cw, cy + ch,
            fill='#1a3a1a', outline=self.DIM_CLR, width=1, tags='curve')

        # Horizontal grid (dB)
        for db in [0, -6, -12, -18, -24, -30]:
            y = self._db_to_y(db)
            c.create_line(cx, y, cx + cw, y,
                          fill=self.DIM_CLR, dash=(2, 6), tags='curve')
            c.create_text(cx - 4, y, text=f"{db}",
                          font=('Courier New', 8), fill=self.DIM_CLR,
                          anchor='e', tags='curve')

        # Vertical grid (frequency)
        for freq in [200, 500, 1000, 2000, 5000, 10000]:
            x = self._freq_to_x(freq)
            c.create_line(x, cy, x, cy + ch,
                          fill=self.DIM_CLR, dash=(2, 6), tags='curve')
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            c.create_text(x, cy + ch + 10, text=label,
                          font=('Courier New', 9), fill=self.DIM_CLR, tags='curve')

        if not self._enabled:
            c.create_line(cx, cy, cx + cw, cy,
                          fill=self.CURVE_CLR, width=2, tags='curve')
            c.create_text(cx + cw // 2, cy + ch // 2, text="FILTER OFF",
                          font=('Courier New', 20, 'bold'), fill='#5a2d2d', tags='curve')
            return

        # Butterworth magnitude response
        points = []
        n_pts = 300
        log_min = math.log10(self.F_MIN)
        log_max = math.log10(self.F_MAX)
        for i in range(n_pts):
            f = 10 ** (log_min + (log_max - log_min) * i / (n_pts - 1))
            mag_sq = 1.0 / (1.0 + (f / self._cutoff) ** (2 * self._order))
            db = 10.0 * math.log10(max(mag_sq, 1e-10))
            x = self._freq_to_x(f)
            y = self._db_to_y(db)
            points.extend([x, y])

        if len(points) >= 4:
            c.create_line(*points, fill=self.CURVE_CLR, width=2,
                          smooth=True, tags='curve')

        # Draggable cutoff dot at -3dB
        dot_x = self._freq_to_x(self._cutoff)
        dot_y = self._db_to_y(-3.01)
        r = self.DOT_R
        c.create_oval(
            dot_x - r, dot_y - r, dot_x + r, dot_y + r,
            fill=self.DOT_CLR, outline='#ffffff', width=2, tags=('curve', 'dot'))

        # Vertical dashed line from dot down
        c.create_line(dot_x, dot_y + r, dot_x, cy + ch,
                      fill=self.DOT_CLR, dash=(3, 3), width=1, tags='curve')

    # -- sliders -----------------------------------------------------------

    def _draw_slider(self, name):
        c = self._canvas
        tag = f'slider_{name}'
        c.delete(tag)

        sx, sw = self.SLIDER_X, self.SLIDER_W
        y = self._cutoff_y if name == 'cutoff' else self._order_y
        frac = self._get_frac(name)
        thumb_x = sx + sw * frac

        # Track
        c.create_rectangle(
            sx, y - self.TRACK_H // 2, sx + sw, y + self.TRACK_H // 2,
            fill=self.SLIDER_TRACK, outline='', tags=tag)
        # Filled portion
        c.create_rectangle(
            sx, y - self.TRACK_H // 2, thumb_x, y + self.TRACK_H // 2,
            fill=self.SLIDER_FILL, outline='', tags=tag)

        # Discrete notches for order
        if name == 'order':
            for val in range(self.ORDER_MIN, self.ORDER_MAX + 1):
                nf = (val - self.ORDER_MIN) / (self.ORDER_MAX - self.ORDER_MIN)
                nx = sx + sw * nf
                c.create_line(nx, y - 12, nx, y + 12,
                              fill=self.DIM_CLR, width=1, tags=tag)
                c.create_text(nx, y + 18, text=str(val),
                              font=('Courier New', 9), fill=self.DIM_CLR, tags=tag)

        # Thumb
        r = self.THUMB_R
        c.create_oval(
            thumb_x - r, y - r, thumb_x + r, y + r,
            fill=self.SLIDER_THUMB, outline=self.LINE_CLR, width=2,
            tags=(tag, f'thumb_{name}'))

    def _get_frac(self, name):
        if name == 'cutoff':
            return (self._cutoff - self.CUTOFF_MIN) / (self.CUTOFF_MAX - self.CUTOFF_MIN)
        return (self._order - self.ORDER_MIN) / (self.ORDER_MAX - self.ORDER_MIN)

    # -- unified canvas event handling -------------------------------------

    def _canvas_press(self, event):
        """Detect what was pressed — button, dot, slider thumb, or slider track."""
        c = self._canvas
        if c is None:
            return "break"

        # Check buttons first
        for (x1, y1, x2, y2, cb) in getattr(self, '_buttons', []):
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                logger.info(f"VoiceEQ button hit at ({event.x},{event.y})")
                # Defer callback — destroying the canvas inside its own
                # event handler can cause Tkinter to silently swallow it.
                self._master.after(1, cb)
                return "break"

        # Check if dot was hit
        items = c.find_overlapping(event.x - 5, event.y - 5, event.x + 5, event.y + 5)
        for item in items:
            tags = c.gettags(item)
            if 'dot' in tags:
                self._dragging = 'dot'
                return "break"
            if 'thumb_cutoff' in tags:
                self._dragging = 'cutoff'
                return "break"
            if 'thumb_order' in tags:
                self._dragging = 'order'
                return "break"

        # Check if tap is on a slider track area (within ±25px of slider Y)
        if abs(event.y - self._cutoff_y) < 25 and self.SLIDER_X <= event.x <= self.SLIDER_X + self.SLIDER_W:
            self._dragging = 'cutoff'
            self._apply_slider(event.x, 'cutoff')
            return "break"
        if abs(event.y - self._order_y) < 25 and self.SLIDER_X <= event.x <= self.SLIDER_X + self.SLIDER_W:
            self._dragging = 'order'
            self._apply_slider(event.x, 'order')
            return "break"

        return "break"

    def _canvas_drag(self, event):
        if not self._dragging:
            return "break"
        if self._dragging == 'dot':
            self._apply_dot_drag(event.x)
        elif self._dragging in ('cutoff', 'order'):
            self._apply_slider(event.x, self._dragging)
        return "break"

    def _canvas_release(self, event):
        self._dragging = None
        return "break"

    def _apply_dot_drag(self, x):
        """Drag the cutoff dot on the curve."""
        if not self._enabled:
            return
        freq = self._x_to_freq(x)
        snapped = round(freq / self.CUTOFF_STEP) * self.CUTOFF_STEP
        snapped = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, snapped))
        if snapped != self._cutoff:
            self._cutoff = snapped
            self._canvas.itemconfig(self._cutoff_val_id, text=f"{self._cutoff} Hz")
            self._draw_slider('cutoff')
            self._draw_curve()

    def _apply_slider(self, x, name):
        """Apply a slider drag/tap position."""
        frac = (x - self.SLIDER_X) / self.SLIDER_W
        frac = max(0.0, min(1.0, frac))
        if name == 'cutoff':
            raw = self.CUTOFF_MIN + (self.CUTOFF_MAX - self.CUTOFF_MIN) * frac
            self._cutoff = round(raw / self.CUTOFF_STEP) * self.CUTOFF_STEP
            self._cutoff = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, self._cutoff))
            self._canvas.itemconfig(self._cutoff_val_id, text=f"{self._cutoff} Hz")
            self._draw_slider('cutoff')
        else:
            raw = self.ORDER_MIN + (self.ORDER_MAX - self.ORDER_MIN) * frac
            self._order = max(self.ORDER_MIN, min(self.ORDER_MAX, round(raw)))
            self._canvas.itemconfig(self._order_val_id, text=str(self._order))
            self._draw_slider('order')
        self._draw_curve()

    # -- buttons -----------------------------------------------------------

    def _draw_button(self, x, y, w, h, text, callback, bg=None):
        bg = bg or self.BTN_BG
        c = self._canvas
        c.create_rectangle(
            x, y, x + w, y + h,
            fill=bg, outline=self.BTN_OUTLINE, width=2)
        c.create_text(
            x + w // 2, y + h // 2, text=text,
            font=('Courier New', 12, 'bold'), fill=self.BTN_TEXT)
        # Store button hitbox for _canvas_press to dispatch
        if not hasattr(self, '_buttons'):
            self._buttons = []
        self._buttons.append((x, y, x + w, y + h, callback))

    def _toggle_enabled(self):
        self._enabled = not self._enabled
        self._build()

    def _preview(self):
        """Synthesize a test phrase with trial settings and play it."""
        def _do_preview():
            try:
                from .tts import get_synthesizer, get_player
                import numpy as np
                synth = get_synthesizer()
                player = get_player()
                if not synth or not player:
                    return

                pcm = synth.synthesize_raw("Hello! BMO is tuning their voice!")
                if not pcm or not self._enabled:
                    if pcm:
                        player.resume()
                        player.play(pcm)
                    return

                from scipy.signal import butter, sosfilt
                sos = butter(self._order, self._cutoff, btype='low',
                             fs=22050, output='sos')
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                filtered = sosfilt(sos, samples)
                pcm = np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()

                player.resume()
                player.play(pcm)
            except Exception as e:
                logger.error(f"VoiceEQ preview error: {e}")

        threading.Thread(target=_do_preview, daemon=True).start()

    def _save(self):
        voice_eq_save(self._enabled, self._cutoff, self._order)
        logger.info(f"Voice EQ saved: enabled={self._enabled}, "
                    f"cutoff={self._cutoff}, order={self._order}")
        self.hide()

    def _cancel(self):
        self.hide()
