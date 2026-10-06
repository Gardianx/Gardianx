# GUI trading automation starter

This one-shot script checks a selected screen region for a reference image, then
can move to and click a configured coordinate. It does not connect to a broker
or decide what to trade.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

Save a small reference image of the indicator or button as `indicator.png`.
Start with detection-only mode:

```powershell
python trading_gui.py --template indicator.png --region 100 100 500 300 --click 800 600
```

Only add `--execute` after validating the screen region, template match, and
coordinate in a paper-trading or other non-live environment:

```powershell
python trading_gui.py --template indicator.png --region 100 100 500 300 --click 800 600 --execute
```

Press **ESC** at any time to stop the script. Mouse movement is broken into
short steps so the kill switch can interrupt it promptly. PyAutoGUI's corner
failsafe remains enabled as an additional safeguard. Actions and dry-run
decisions are appended to `gui_actions.log`.

The click occurs only when the template is detected and `--execute` is present.
Use a distinctive, stable template and verify the coordinates on your display;
screen scaling, window movement, or layout changes can make fixed coordinates
unsafe. This starter is not a substitute for broker-side safeguards.
