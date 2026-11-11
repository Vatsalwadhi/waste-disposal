# Waste Disposal Coach

Point a camera at an item and get told which bin it goes in, and why.

    detect item (YOLO-World)  ->  food-residue check (OpenCV)  ->  local rule (rules.json)  ->  on-screen advice

## Run

    pip install -r requirements.txt
    python test_coach.py                     # rules + residue checks, no camera needed
    python waste_coach.py                    # webcam, generic rules
    python waste_coach.py --region india     # India four-stream profile
    python waste_coach.py --source photo.jpg --save out.jpg

Keys: `q` quit, `r` cycle region, `c` force "contaminated", `s` snapshot.

Model weights download on first run (needs internet once). Run it on Windows
or a machine with a camera; WSL does not see the webcam without extra setup.

## How it works

- **Detection.** YOLO-World is open-vocabulary: the class list is the `prompts`
  in `rules.json` ("aluminum can", "pizza box", ...), so cans, cardboard and
  wrappers work with no training. `--model yolov8n.pt` falls back to a plain
  COCO model, which only knows bottles, cups, bowls, books and fruit.
- **Contamination.** `residue_score()` finds the item's base colour (k-means in
  LAB), then counts pixels that differ from it and look like food or grease
  (red-to-yellow hue, more saturated or darker than the base). Above 6% of the
  area, the item is treated as contaminated and the stain is tinted red.
  It only runs on items with `"check_residue": true`.
- **Local rules decide.** Each region lists only what differs from `default`.
  The same greasy pizza box is general waste in `default`, compost in
  `organics_city`, and non-recyclable dry waste in `india`.

## Add your own city

Copy a region block in `rules.json`, rename the bins, and override the items
your municipality treats differently. To add a new item, add it under `items`
with detector prompts and give it a rule in `default`.

## Limits

- The residue check is a colour heuristic. Red or brown printing fools it
  (which is why it is off for bottles and cans), and it cannot see clear
  liquids or residue inside a container. A small classifier trained on
  clean/dirty crops is the natural upgrade.
- The regional profiles are illustrative. Check them against your local
  body's published list.
