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
