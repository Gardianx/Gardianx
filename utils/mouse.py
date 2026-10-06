"""Mouse input wrapper that defaults to dry-run and requires explicit opt-in."""

from typing import Any, Callable


class MouseController:
    def __init__(
        self,
        *,
        pyautogui_module: Any | None = None,
        execute: bool = False,
        check_safety: Callable[[], None] | None = None,
    ) -> None:
        if pyautogui_module is None:
            import pyautogui as pyautogui_module
        self._mouse = pyautogui_module
        self._execute = execute
        self._check_safety = check_safety or (lambda: None)

    @property
    def execute_enabled(self) -> bool:
        return self._execute

    def click(self, x: int, y: int) -> None:
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError("x must be an integer screen coordinate.")
        if isinstance(y, bool) or not isinstance(y, int):
            raise TypeError("y must be an integer screen coordinate.")
        self._check_safety()
        if not self._execute:
            print(f"DRY RUN: would click at ({x}, {y})")
            return
        self._mouse.click(x, y)

    def replace_text(self, x: int, y: int, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input must be a non-empty string.")
        self._check_safety()
        if not self._execute:
            print(f"DRY RUN: would replace field at ({x}, {y}) with {text!r}")
            return
        self._mouse.click(x, y)
        self._check_safety()
        self._mouse.hotkey("ctrl", "a")
        self._mouse.write(text)
