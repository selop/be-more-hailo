# core/timer_clock.py — countdown clock shown while a timer runs
"""A small round clock in the top-left corner: a ring that empties as the
timer runs down, with the remaining time in the middle. With several timers
it shows the one that ends first and a "+N" badge for the others.

The widget polls ``get_timers`` from Tk's own loop, so it needs no calls
from worker threads.
"""

import tkinter as tk

from .config import CHARACTER
from .timers import format_clock


class TimerClock:
    """Countdown overlay.

    Parameters
    ----------
    master : tk.Tk
    get_timers : callable() -> list[dict]
        Running timers, soonest first, each with ``remaining_s`` and
        ``total_s`` (see ``Conversation.active_timers``).
    is_hidden : callable() -> bool or None
        Returns True while something else owns the screen (a photo, the
        voice tuner) and the clock should stay out of the way.
    """

    # Layout
    X, Y = 18, 18
    SIZE = 104
    RING_W = 9
    PAD = 9
    INTERVAL_MS = 200
    URGENT_S = 10           # the ring turns to the alert colour for the last seconds

    FONT_FAMILY = 'Courier New'

    def __init__(self, master, get_timers, is_hidden=None):
        self._master = master
        self._get_timers = get_timers
        self._is_hidden = is_hidden
        theme = CHARACTER.theme
        self._bg = theme["face_bg"]
        self._fill = theme["bubble_fill"]
        self._ink = theme["status_fg"]
        self._track = theme["bubble_outline"]
        self._alert = theme["alert_fg"]
        self._canvas = None
        self._visible = False
        self._tick()

    # -- drawing -----------------------------------------------------------

    def _build(self):
        s, p = self.SIZE, self.PAD
        c = tk.Canvas(self._master, width=s, height=s, bg=self._bg, highlightthickness=0, bd=0)
        # Dial
        c.create_oval(2, 2, s - 2, s - 2, fill=self._fill, outline=self._track, width=2)
        # Ring: a faint full track with the remaining part drawn over it
        c.create_oval(p, p, s - p, s - p, outline=self._track, width=2)
        self._arc = c.create_arc(p, p, s - p, s - p, start=90, extent=-359.9,
                                 style=tk.ARC, outline=self._ink, width=self.RING_W)
        self._text = c.create_text(s // 2, s // 2, text="", fill=self._ink,
                                   font=(self.FONT_FAMILY, 20, 'bold'))
        self._more = c.create_text(s // 2, s // 2 + 22, text="", fill=self._track,
                                   font=(self.FONT_FAMILY, 10, 'bold'))
        self._canvas = c

    def _draw(self, timers):
        first = timers[0]
        remaining = max(first["remaining_s"], 0.0)
        total = max(first["total_s"], 0.001)
        fraction = min(remaining / total, 1.0)
        urgent = remaining <= self.URGENT_S
        # In the last seconds the ring blinks between the alert colour and the track
        if urgent and int(remaining * 2) % 2 == 0:
            colour = self._track
        else:
            colour = self._alert if urgent else self._ink
        c = self._canvas
        c.itemconfig(self._arc, extent=-max(359.9 * fraction, 0.1), outline=colour)
        label = format_clock(remaining)
        c.itemconfig(self._text, text=label, fill=self._alert if urgent else self._ink,
                     font=(self.FONT_FAMILY, 20 if len(label) <= 5 else 14, 'bold'))
        c.itemconfig(self._more, text=f"+{len(timers) - 1}" if len(timers) > 1 else "")

    # -- loop --------------------------------------------------------------

    def _tick(self):
        try:
            timers = self._get_timers()
            hidden = bool(self._is_hidden and self._is_hidden())
            if timers and not hidden:
                if self._canvas is None:
                    self._build()
                self._draw(timers)
                if not self._visible:
                    self._visible = True
                    self._canvas.place(x=self.X, y=self.Y)
                    # Raise the widget above the face. Not self._canvas.lift():
                    # on a Canvas that name means "raise a canvas item" and
                    # fails without arguments.
                    tk.Misc.tkraise(self._canvas)
            elif self._visible:
                self._canvas.place_forget()
                self._visible = False
        finally:
            self._master.after(self.INTERVAL_MS, self._tick)
