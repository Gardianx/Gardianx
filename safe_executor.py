"""Mouse execution with an ESC kill switch and an append-only action log."""

from datetime import datetime
import math
from pathlib import Path
import time


class EmergencyStop(RuntimeError):
    """Raised when the user presses ESC."""


class SafeExecutor:
    _MAX_MOVE_STEP_SECONDS = 0.02

    def __init__(
        self,
        log_path: str | Path = "gui_actions.log",
        *,
        pyautogui_module=None,
        listener_factory=None,
        esc_key=None,
    ) -> None:
        if pyautogui_module is None:
            import pyautogui as pyautogui_module
        self._pyautogui = pyautogui_module
        self._log_path = Path(log_path)
        self._listener_factory = listener_factory
        self._esc_key = esc_key
        self._listener = None
        self._stop_requested = False
        self._previous_pause = None

    def __enter__(self):
        if self._listener_factory is None or self._esc_key is None:
            from pynput import keyboard

            if self._listener_factory is None:
                self._listener_factory = keyboard.Listener
            if self._esc_key is None:
                self._esc_key = keyboard.Key.esc

        self._listener = self._listener_factory(on_press=self._on_press)
        self._listener.start()
        self._previous_pause = self._pyautogui.PAUSE
        self._pyautogui.PAUSE = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener.join()
        if self._previous_pause is not None:
            self._pyautogui.PAUSE = self._previous_pause

    def _on_press(self, key):
        if key == self._esc_key:
            self._stop_requested = True
            return False
        return None

    def check_kill_switch(self) -> None:
        if self._stop_requested:
            raise EmergencyStop("ESC pressed; stopping GUI automation.")

    def log_action(self, action: str, x: int, y: int) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} | {action} | x={x} y={y}\n")

    def move_and_click(self, x: int, y: int, duration: float = 0.25) -> None:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Move duration must be a finite, non-negative number.")
        self.check_kill_switch()
        self.log_action("MOVE_START", x, y)

        try:
            start_x, start_y = self._pyautogui.position()
            steps = max(1, math.ceil(duration / self._MAX_MOVE_STEP_SECONDS))
            for step in range(1, steps + 1):
                self.check_kill_switch()
                progress = step / steps
                next_x = round(start_x + (x - start_x) * progress)
                next_y = round(start_y + (y - start_y) * progress)
                self._pyautogui.moveTo(next_x, next_y, duration=0)
                if duration:
                    time.sleep(duration / steps)
            self.check_kill_switch()
        except EmergencyStop:
            self.log_action("MOVE_INTERRUPTED", x, y)
            raise

        self.log_action("MOVE_COMPLETE", x, y)
        self.check_kill_switch()
        self.log_action("CLICK_START", x, y)
        self._pyautogui.click(x, y)
        self.log_action("CLICK_COMPLETE", x, y)
