"""Waste Disposal Coach: point a camera at an item, get disposal instructions.

Pipeline:  detect item (YOLO)  ->  check for food residue (OpenCV)
           ->  look up the local rule (rules.json)  ->  draw the advice.

Run:   python waste_coach.py                       # webcam, default rules
       python waste_coach.py --region india
       python waste_coach.py --source photo.jpg --save out.jpg
Keys:  q quit | r next region | c force "contaminated" on/off | s snapshot
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"recyclable": (200, 140, 0), "compostable": (60, 170, 60), "general": (90, 90, 90)}  # BGR
RESIDUE_THRESHOLD = 0.06   # share of the item's area that must look stained


# ---------------------------------------------------------------- rules ----
class Rules:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.items = data["items"]
        self.regions = data["regions"]
        # detector class name -> item key, for both open-vocabulary and COCO models
        self.name_to_item = {}
        for key, item in self.items.items():
            for name in item["prompts"] + item.get("coco", []):
                self.name_to_item[name.lower()] = key

    def prompts(self):
        return [p for item in self.items.values() for p in item["prompts"]]

    def decide(self, item_key, contaminated, region):
        """Local rule first, then the default rule. Returns a dict for display."""
        local, default = self.regions[region], self.regions["default"]
        rule = local["rules"].get(item_key) or default["rules"][item_key]
        dirty = contaminated and "contaminated" in rule
        verdict = rule["contaminated" if dirty else "clean"]
        category = verdict["category"]
        bins = {**default["bins"], **local.get("bins", {})}
        return {
            "item": self.items[item_key]["label"],
            "category": category,
            "bin": verdict.get("bin", bins[category]),
            "tip": verdict["tip"],
            "contaminated": dirty,
        }


# -------------------------------------------------------- contamination ----
def residue_score(crop_bgr):
    """Estimate the share of an item's surface covered by food residue.

    Heuristic, not a trained model: find the item's base colour, then count
    pixels that differ from it AND look like food or grease (red-to-yellow
    hue, clearly more saturated or clearly darker than the base).
    Returns (score 0..1, uint8 mask the size of the crop).
    """
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return 0.0, np.zeros((h, w), np.uint8)
    scale = 160 / max(h, w)
    small = cv2.resize(crop_bgr, (max(8, int(w * scale)), max(8, int(h * scale))))
    small = cv2.GaussianBlur(small, (5, 5), 0)
    sh, sw = small.shape[:2]

    # Look at the centre ellipse only: box corners are mostly background.
    roi = np.zeros((sh, sw), np.uint8)
    cv2.ellipse(roi, (sw // 2, sh // 2), (int(sw * 0.42), int(sh * 0.42)), 0, 0, 360, 255, -1)
    inside = roi > 0

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Base material = biggest of 3 colour clusters.
    pixels = lab[inside]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    base_idx = np.bincount(labels.ravel(), minlength=3).argmax()
    base_lab = centers[base_idx]
    base_hsv = np.median(hsv[inside][labels.ravel() == base_idx], axis=0)

    dist = np.linalg.norm(lab - base_lab, axis=2)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    food_hue = (hue <= 30) | (hue >= 170)                 # reds, oranges, browns, yellows
    stained = (sat > base_hsv[1] + 35) | (val < base_hsv[2] - 45)
    not_glare = (val > 35) & ~((val > 225) & (sat < 40))
    mask = (inside & (dist > 20) & food_hue & stained & not_glare & (sat > 50)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    score = float(mask.sum() / 255) / max(1, int(inside.sum()))
    return score, cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------- detector ----
class Detector:
    def __init__(self, weights, rules, conf):
        from ultralytics import YOLO, YOLOWorld
        self.rules, self.conf = rules, conf
        if "world" in Path(weights).name.lower():
            self.model = YOLOWorld(weights)
            self.model.set_classes(rules.prompts())      # open vocabulary: cans, cardboard, wrappers...
        else:
            self.model = YOLO(weights)                   # plain COCO model: bottles, cups, fruit only

    def __call__(self, frame):
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        found = []
        for box in result.boxes:
            name = result.names[int(box.cls)].lower()
            item = self.rules.name_to_item.get(name)
            if item:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                found.append({"item": item, "conf": float(box.conf), "box": (x1, y1, x2, y2)})
        return found


# -------------------------------------------------------------- drawing ----
def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    return lines + [line] if line else lines


def annotate(frame, detections, rules, region, smooth, force_dirty):
    """Draw boxes and the advice panel. `smooth` holds running residue scores."""
    out = frame.copy()
    best = None
    for det in sorted(detections, key=lambda d: d["conf"]):
        x1, y1, x2, y2 = det["box"]
        key = det["item"]
        score = 0.0
        if rules.items[key]["check_residue"]:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            raw, mask = residue_score(crop)
            score = smooth[key] = 0.6 * smooth.get(key, raw) + 0.4 * raw   # steadier on video
            if score >= RESIDUE_THRESHOLD and mask.shape[:2] == crop.shape[:2]:
                view = out[max(0, y1):y2, max(0, x1):x2]
                view[mask > 0] = (0.5 * view[mask > 0] + (0, 0, 127)).astype(np.uint8)
        dirty = force_dirty or score >= RESIDUE_THRESHOLD
        advice = rules.decide(key, dirty, region)
        advice["score"] = score
        color = COLORS[advice["category"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{advice['item']} -> {advice['bin']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y1 - 6 if y1 > th + 10 else y1 + th + 6
        cv2.rectangle(out, (x1, ty - th - 5), (x1 + tw + 8, ty + 5), color, -1)
        cv2.putText(out, tag, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        best = advice                                   # last = most confident

    h, w = out.shape[:2]
    lines = [f"Rules: {rules.regions[region]['name']}"]
    if best:
        state = "  [food residue detected]" if best["contaminated"] else ""
        lines.append(f"{best['item']}: {best['category'].upper()} -> {best['bin']}{state}")
        lines += wrap(best["tip"], max(30, w // 11))
    else:
        lines.append("Show me a bottle, can, box or wrapper...")
    panel_h = 22 * len(lines) + 12
    out[h - panel_h:] = (out[h - panel_h:] * 0.25).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, h - panel_h + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255) if i else (180, 180, 180), 1, cv2.LINE_AA)
    return out, best


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description="Waste Disposal Coach")
    ap.add_argument("--source", default="0", help="webcam index, or path to an image/video")
    ap.add_argument("--region", default="default", help="rules profile in rules.json")
    ap.add_argument("--rules", default=str(HERE / "rules.json"))
    ap.add_argument("--model", default="yolov8s-world.pt",
                    help="YOLO-World weights (recommended) or a COCO model such as yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--save", help="write the annotated image here (image sources)")
    args = ap.parse_args()

    rules = Rules(args.rules)
    regions = list(rules.regions)
    if args.region not in regions:
        ap.error(f"unknown region '{args.region}'. Choose from: {', '.join(regions)}")
    region = args.region
    detector = Detector(args.model, rules, args.conf)
    smooth, force_dirty = {}, False

    if Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(args.source)
        if frame is None:
            ap.error(f"cannot read image {args.source}")
        out, best = annotate(frame, detector(frame), rules, region, smooth, force_dirty)
        print(json.dumps(best, indent=2) if best else "No known item detected.")
        if args.save:
            cv2.imwrite(args.save, out)
        else:
            cv2.imshow("Waste Disposal Coach", out)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        ap.error(f"cannot open source {args.source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, _ = annotate(frame, detector(frame), rules, region, smooth, force_dirty)
        if force_dirty:
            cv2.putText(out, "FORCED: contaminated", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Waste Disposal Coach", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            region = regions[(regions.index(region) + 1) % len(regions)]
        if key == ord("c"):
            force_dirty = not force_dirty
        if key == ord("s"):
            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", out)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
