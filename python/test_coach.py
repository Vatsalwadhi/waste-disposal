"""Checks for the parts that need no camera or model:  python test_coach.py"""
import cv2
import numpy as np

from waste_coach import HERE, RESIDUE_THRESHOLD, Rules, annotate, residue_score

rules = Rules(HERE / "rules.json")

# Every item has a default rule, and every region rule points at a real item and bin.
for key in rules.items:
    assert key in rules.regions["default"]["rules"], key
for region, spec in rules.regions.items():
    for key, rule in spec["rules"].items():
        assert key in rules.items, (region, key)
        for state in rule:
            rules.decide(key, state == "contaminated", region)

# Same greasy box, three different local answers.
assert rules.decide("cardboard", False, "default")["category"] == "recyclable"
assert rules.decide("cardboard", True, "default")["category"] == "general"
assert rules.decide("cardboard", True, "organics_city")["category"] == "compostable"
assert rules.decide("cardboard", True, "india")["bin"].startswith("Dry waste")
assert rules.decide("plastic_bottle", True, "india")["bin"] == "Dry waste bin"   # no contaminated rule -> clean rule

# Residue heuristic on synthetic cardboard.
rng = np.random.default_rng(0)
clean = np.full((240, 240, 3), (110, 150, 185), np.uint8)                 # kraft brown, BGR
clean = cv2.add(clean, rng.integers(0, 12, clean.shape, dtype=np.uint8))
greasy = clean.copy()
cv2.circle(greasy, (100, 110), 38, (40, 80, 150), -1)                     # dark grease patch
cv2.circle(greasy, (150, 150), 22, (30, 60, 200), -1)                     # tomato sauce
s_clean, _ = residue_score(clean)
s_greasy, mask = residue_score(greasy)
print(f"residue score  clean={s_clean:.3f}  greasy={s_greasy:.3f}  threshold={RESIDUE_THRESHOLD}")
assert s_clean < RESIDUE_THRESHOLD < s_greasy
assert mask[110, 100] and not mask[30, 120]

# Drawing path end to end with a fake detection.
frame = np.full((480, 640, 3), 230, np.uint8)
frame[120:360, 200:440] = greasy
out, best = annotate(frame, [{"item": "cardboard", "conf": 0.9, "box": (200, 120, 440, 360)}],
                     rules, "default", {}, False)
assert best["contaminated"] and best["category"] == "general"
cv2.imwrite(str(HERE / "demo_overlay.jpg"), out)
print("all checks passed")
