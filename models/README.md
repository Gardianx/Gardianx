# Model weights

Place the UI-trained Ultralytics checkpoint at `models/best.pt`, or pass a
different checkpoint path to `main.py --model`.

The generic `yolov8n.pt` checkpoint is not trained to identify broker-specific
tabs, chart states, assets, or trading controls. The broker-aware runner needs
a UI-trained checkpoint with the expected labels; it defaults to dry-run, and
live PyAutoGUI input requires explicit `--execute` and an explicit one-shot
action.
