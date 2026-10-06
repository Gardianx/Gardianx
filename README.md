# GUI trading automation starter

This one-shot script checks a selected screen region for a reference image, then
can move to and click a configured coordinate. It does not connect to a broker
or decide what to trade.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Crop button templates

Keep the trading interface visible, then run:

```powershell
python template_cropper.py
```

The script captures the full screen once. For each selection window, drag a
rectangle around the Buy button and press **ENTER** to save it; then select the
Sell button and press **ENTER**. Press **C** to skip either crop. Saved images
go into the project's `templates` folder as `buy_button.png` and
`sell_button.png`. A new successful crop replaces an image with the same name.

To capture the calibrated button crops automatically after switching to
Firefox, run:

```powershell
python capture_with_delay.py
```

The script prints a five-second countdown, captures the full screen, and saves
60x30 crops centered at the calibrated Buy `(1188, 23)` and Sell `(1308, 21)`
button coordinates.

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

## Find screen coordinates

Run `python coordinate_finder.py`, move the mouse to a point on the screen, and
press **F8** to print its exact cursor coordinates in the terminal. Repeat for
chart boundaries and button positions; press **ESC** to stop the utility.

## Stateful Buy/Sell logic

`trading_logic.py` provides a `TradingLogic` controller. Give it separate
`ImageDetector` instances for the Buy and Sell templates plus a `SafeExecutor`.
Each call to `step()` checks only Buy while not holding, or only Sell while
holding. A detected button is clicked at the center of its matched image.
Successful clicks flip `IS_HOLDING`, and the default three-second cooldown
prevents another scan-and-click action until it expires.

The position state exists only in memory and starts as not holding each time
the process starts. It represents clicks made by this controller, not broker
order fills or the actual account position. Reconcile account state separately
before using this with a live trading interface.

## Run the master agent

After using `template_cropper.py` to save both button templates and installing
the PyTorch/torchvision builds from `requirements-training.txt`, run:

```powershell
python main_agent.py
```

The default chart bounds are top-left `(2, 0)` to bottom-right `(1362, 728)`,
represented as the screen region `(left=2, top=0, width=1360, height=728)`.
Each scan crops that region from one desktop screenshot and passes the crop to
`model_brain.py` for inference; template checks use the same captured crop.
Buy and Sell clicks target `(1188, 23)` and `(1308, 21)` respectively, but are
still gated by a matching button template, position state, and cooldown. The
`--region` option can override the chart region. The agent waits 1.5 seconds
between scans; use `--interval` to set a delay from 1 to 2 seconds and
`--cooldown` to change the minimum time between clicks. By default, it runs in
dry-run mode:
it logs matching actions and updates its simulated in-memory state without
moving or clicking the mouse. Add `--execute` only after validating the
templates and coordinates in a paper-trading or other non-live environment.
Press **ESC** to interrupt the wait or stop before another movement/click.

The ESC listener runs on `pynput`'s background listener thread. Mouse movement
checks the stop signal in short steps; no software listener can guarantee
microsecond-level interruption of an operating-system or GUI call.

The master loop passes the chart crop to `TradingBrain.predict_signal()` and
uses the CNN checkpoint's highest-confidence `buy`, `sell`, or `hold`
prediction. Each scan appends a result to `trade_history.csv` through
`performance_logger.py`. The CNN classifies chart pixels; no market data or
live asset price is currently collected.

## Trade history CSV

`performance_logger.py` exposes `log_trade(action, price, confidence_score)`.
It creates `trade_history.csv` beside the module when needed, writes a header,
and appends timestamped `BUY`, `SELL`, or `HOLD` records. The price field
accepts either a number or an asset-state description when a price is
unavailable. Confidence must be a number from 0 to 1.

## Model inference

`model_brain.py` loads `trading_vision_model.pth` from the project root,
builds a three-class torchvision ResNet-18, loads its raw or wrapped state
dictionary with validated key compatibility, resizes each chart crop to
224x224, and returns the highest-confidence prediction. For raw state
dictionaries without class metadata, output indices map alphabetically to
`buy`, `hold`, and `sell`, matching `ImageFolder` conventions. Install the
platform-appropriate PyTorch and torchvision packages from
`requirements-training.txt` before running `main_agent.py`. Model fitting is
handled by `train_vision_model.py`; `TradingBrain.train_model()` is not
implemented.

## YOLO screen-object detection starter

Install Ultralytics with `python -m pip install ultralytics` (also listed in
`requirements-training.txt`) and run:

```powershell
python yolo_detector.py
```

The starter loads the generic `yolov8n.pt` model, captures the full screen,
then reports each detected class, bounding box, center, and confidence. The
default model is pretrained for general-purpose object classes, not browser
tabs, chart controls, or trading buttons; detecting those UI elements requires
a YOLO model trained on labeled screenshots.

## Modular broker and asset scaffold

The broker-aware, state-monitoring loop can be started with:

```powershell
python main.py
```

`core/screen_capture.py` captures the desktop, `core/detection.py` loads
`models/best.pt`, `core/vision.py` finds controls and OCRs the active asset
label, and `core/state.py` identifies the broker and distinguishes standard
EUR/USD from EUR/USD OTC. `brokers/exness.py` and
`brokers/pocket_option.py` fill detected lot/investment/expiration controls and
click detected action buttons. `utils/mouse.py` is dry-run by default. The
central thresholds and Exness/OTC rejection policy are in `config.py`.
The UI-trained checkpoint must expose broker labels (`exness` or
`pocket_option`), `active_asset_label`, broker control labels (`buy_button`,
`sell_button`, `lot_size_input`, `call_button`, `put_button`,
`investment_input`, `expiration_input`), and safety popup labels
(`insufficient_balance`, `market_closed`) for the workflows being used.
Pocket Option annotations can use `pocket_buy`, `pocket_sell`, `pocket_amount`,
and `pocket_time`; those labels are accepted in place of the generic action
buttons, investment field, and expiration field. `pocket_payout`,
`pocket_trades`, `pocket_wallet`, `pocket_candle_timer`,
`pocket_candle_time`, `bullish_otc_candle`, and `bearish_otc_candle` are
recognized as Pocket Option UI detections for future strategy logic. The
current runner does not derive a trade signal from the candle detections.
For the selected browser tab, label its bounding box as `active_exness_tab` or
`active_pocket_option_tab`; the inactive visible tab should not receive either
active-tab label. These explicit selected-tab detections take precedence over
broker logo/control detections when resolving broker state.

The agent continuously monitors the screen and accepts an explicit one-shot
external action; it never treats a visible button as an instruction to trade.
For example, this validates a requested Exness buy against live detections but
does not send mouse input:

```powershell
python main.py --action buy --asset "EUR/USD" --amount 0.01
```

For Pocket Option, supply an investment amount and expiration:

```powershell
python main.py --action call --asset "EUR/USD OTC" --amount 5 --expiration-seconds 60
```

Actual PyAutoGUI input requires the additional `--execute` opt-in. The agent
fails closed and pauses on unknown/conflicting broker identity, unverified or
mismatched OCR asset, an Exness/OTC mismatch, or recognized
`insufficient_balance`/`market_closed` popup detections. Press **ESC** to stop.
OCR requires `pytesseract` (in `requirements.txt`) and the separate Tesseract
OCR executable installed on the system. Label names used by `best.pt` must
match the configured aliases/control labels in `config.py` and the broker
adapters. The weight file is local and is not committed.

### Train the broker UI detector

The detector identifies screen regions; it does not learn a trading strategy.
To train it, collect broker screenshots and label each visible object with a
bounding box using the exact class names and order in
[`dataset/ui/data.yaml`](./dataset/ui/data.yaml). Keep screenshots from the
same capture sessions together when splitting train and validation data to
reduce near-duplicate leakage. Include varied screen sizes, themes, chart
states, and both popup and no-popup examples. Store YOLO-format annotations
under `dataset/ui/labels/train` and `dataset/ui/labels/val`, paired with images
under `dataset/ui/images/train` and `dataset/ui/images/val`.

Install the packages in `requirements-training.txt` and run the training
script from the repository root:

```powershell
python train_ui_detector.py --data dataset/ui/data.yaml --device 0
```

On Kaggle, upload the dataset as an input and pass its `data.yaml` path. Use
`--device cpu` if a GPU is unavailable, or pass `--base-model` if the starting
checkpoint is provided as a Kaggle input. The script validates that dataset
class names match the application's UI labels, trains and validates YOLO, and
copies the best checkpoint to `models/best.pt`. Validate detections on held-out
screenshots before enabling live input. Trading decisions remain separate and
must come from a strategy you define; the current orchestrator does not infer
trades from detections.

For Pocket Option asset identity, label the rectangle containing the currently
selected symbol as `active_asset_label`; OCR reads its text. The supported OTC
symbols are EUR/USD, AED/CNY, AUD/NZD, EUR/NZD, CAD/CHF, USD/JPY, EUR/RUB,
GBP/JPY, and GBP/CAD. Avoid separate YOLO classes for every symbol: the same
symbol may appear in the open asset menu, and the detector alone cannot tell
which occurrence is selected.

The browser tab strip usually shows both Exness and Pocket Option tabs at once.
Do not label both as active: annotate only the selected tab with its
`active_*_tab` class. Otherwise the detector would know both tabs exist but
could not determine which broker is currently open.

## Prepare dataset folders

Run `python setup_dataset.py` to create `dataset/train` and
`dataset/validation`, each with `buy`, `sell`, and `hold` subfolders. The
command is safe to rerun and prints the dataset location when complete.

## Download labeled chart images from Hugging Face

Install the optional Hugging Face `datasets` package through
`requirements-training.txt`, then run:

```powershell
python huggingface_ingestion.py
```

The loader downloads up to 1,500 images from
`StephanAkkerman/stock-charts` and saves them under the separate
`dataset/chart_recognition/` tree. The source labels are `charts` and
`non-charts`, so the loader preserves them and does not fabricate BUY/SELL/HOLD
labels. It writes a stratified validation sample (every fifth image within
each source class) and places the remaining images in that class's training
folder. This binary chart-recognition dataset is separate from the trading
signal dataset.

Train the matching binary classifier with:

```powershell
python train_chart_classifier.py
```

## Preprocess and label chart images

Put chart screenshots in `raw_charts/`, then run:

```powershell
python dataset_preprocessor.py
```

For each supported image, drag a box around the chart candles/price action in
the OpenCV window and press **ENTER** or **SPACE** to accept; press **C** to
skip that image. Choose a `buy`, `sell`, or `hold` label and a `train` or
`validation` split in the terminal. The selected crop is resized to 224x224
and saved in the matching dataset folder. Original images in `raw_charts/` are
left unchanged. If a destination filename already exists, a numbered filename
is used instead of overwriting it.

## Train the vision CNN

Install PyTorch and torchvision builds appropriate for your computer
(CPU-only or CUDA) with:

```powershell
python -m pip install -r requirements-training.txt
```

After collecting labeled images in every class for both dataset splits, run:

```powershell
python train_vision_model.py --epochs 15
```

The script loads the `dataset/train/` and `dataset/validation/` folders with
`torchvision.datasets.ImageFolder`, resizes images to 224x224, and trains a
three-class CNN. It prints per-epoch loss and accuracy and writes model weights
and class-index metadata to `trading_vision_model.pth`. Override the epoch
count, batch size, learning rate, dataset paths, output path, or device with
the corresponding command-line options. The existing dataset folders are
currently just empty structure; add labeled images to each class before
training.
