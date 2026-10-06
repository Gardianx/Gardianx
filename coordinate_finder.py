"""Print the current mouse coordinates whenever F8 is pressed."""

from typing import Callable


def run_coordinate_finder(
    *,
    keyboard_module=None,
    mouse_controller_factory: Callable | None = None,
    report: Callable[[str], None] = print,
) -> None:
    """Listen for F8 in the background and report the current cursor position."""
    if keyboard_module is None or mouse_controller_factory is None:
        from pynput import keyboard, mouse

        if keyboard_module is None:
            keyboard_module = keyboard
        if mouse_controller_factory is None:
            mouse_controller_factory = mouse.Controller

    mouse_controller = mouse_controller_factory()

    def on_press(key):
        if key == keyboard_module.Key.f8:
            x, y = mouse_controller.position
            report(f"Mouse position: X={x}, Y={y}")
        elif key == keyboard_module.Key.esc:
            report("Stopping coordinate finder.")
            return False
        return None

    report("Coordinate finder is running.")
    report("Move your mouse anywhere on the screen and press F8 to print its coordinates.")
    report("Press ESC to stop.")
    with keyboard_module.Listener(on_press=on_press) as listener:
        listener.join()


def main() -> int:
    run_coordinate_finder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
