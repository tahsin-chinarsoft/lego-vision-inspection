"""
ChinarSoft - LEGO Assembly Line Vision Inspection
Two stations with classic machine vision (OpenCV, no AI training):
  Station 1 - KIT CHECK      : are all parts in the tray? (count by color)
  Station 2 - FINAL ASSEMBLY : is every part in the right place? (16-point grid check)

Keys:
  1 / 2   switch station          SPACE  trigger: inspect + log + save image
  a       auto mode on/off        m      show color masks (debug)
  g       show/hide grid          r      reset counters          q / ESC  quit
  UP / DOWN arrow (or + / -)   inspection box bigger / smaller
  f       full screen on/off (for recording)
  i j k l move inspection box (up / left / down / right)   c  re-center box

Run:
  python3 lego_inspect.py                 # camera 0
  python3 lego_inspect.py --camera 1      # other camera
  python3 lego_inspect.py --image samples/kit_ok.jpg --station 1 --roi 0.85   # test on a photo
"""
import argparse, csv, os, time
from datetime import datetime
import cv2
import numpy as np

# ---------------------------------------------------------------- CONFIG
# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255). Tune these for your light.
COLORS = {
    "ORANGE": [((11, 120, 45), (28, 255, 255))],
    "RED":    [((0, 120, 26), (10, 255, 255)), ((165, 120, 26), (179, 255, 255))],
    "LIME":   [((30, 80, 30), (50, 255, 255))],
    "BLACK":  [((0, 0, 0), (179, 255, 25)), ((0, 0, 0), (179, 110, 55))],
}
DRAW = {"ORANGE": (0, 140, 255), "RED": (40, 40, 255), "LIME": (60, 255, 120),
        "BLACK": (255, 80, 255), "UNKNOWN": (255, 255, 0)}

# Station 1: Bill of Materials (color -> expected count)
BOM = {"ORANGE": 1, "RED": 4, "BLACK": 1, "LIME": 1}
BOM_NAMES = {"ORANGE": "Orange slope base 4x4", "RED": "Trans-red round 1x1",
             "BLACK": "Black brick 2x2", "LIME": "Lime round 1x1"}

# Station 2: stud grid position on the base (fractions of the base rectangle)
GRID_X = [0.19, 0.42, 0.65, 0.88]
GRID_Y = [0.195, 0.36, 0.525, 0.71]
CELL_R = 0.065                      # ROI radius as fraction of base width
# Expected color per stud: R=red, O=orange, C=center (black brick area)
EXPECTED = ["R O O R",
            "O C C O",
            "O C C O",
            "R O O R"]
PROC_WIDTH = 960                    # processing resolution (speed)
MIN_PART_AREA = 0.0008              # min part area (fraction of frame)
VERSION = "v1.4 (ROI box + arrows + window)"
LOG_DIR = "inspection_log"
# Inspection box (ROI): a square in the middle of the image. Only objects inside it are checked.
ROI_SIZE = 0.60                     # side length as fraction of the image height
ROI_STEP = 0.05                     # +/- step
# Arrow key codes returned by cv2.waitKeyEx (macOS, Linux, Windows)
KEY_UP = (63232, 65362, 2490368)
KEY_DOWN = (63233, 65364, 2621440)
ROI_MOVE = 0.02                     # i/j/k/l step (fraction of image width/height)

# ---------------------------------------------------------------- VISION
def color_mask(hsv, name):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[name]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return m

def object_mask(hsv):
    """Everything that is not the white/grey background."""
    m = ((hsv[..., 1] > 90) | (hsv[..., 2] < 70)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m

def dominant_color(hsv, mask=None):
    total = cv2.countNonZero(mask) if mask is not None else hsv.shape[0] * hsv.shape[1]
    best, best_f = "UNKNOWN", 0.0
    if total == 0:
        return best, 0.0
    for name in COLORS:
        cm = color_mask(hsv, name)
        if mask is not None:
            cm = cv2.bitwise_and(cm, mask)
        f = cv2.countNonZero(cm) / total
        if f > best_f:
            best, best_f = name, f
    return (best if best_f >= 0.35 else "UNKNOWN"), best_f

# ---- Station 1: kit check
def touches_border(c, shape, margin=3):
    x, y, w, h = cv2.boundingRect(c)
    H, W = shape[:2]
    return x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin

def inspect_kit(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    om = object_mask(hsv)
    om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = MIN_PART_AREA * img.shape[0] * img.shape[1]
    counts = {k: 0 for k in BOM}
    counts["UNKNOWN"] = 0
    parts = []
    for c in cs:
        if cv2.contourArea(c) < min_a or touches_border(c, om.shape):
            continue
        pm = np.zeros(om.shape, np.uint8)
        cv2.drawContours(pm, [c], -1, 255, -1)
        name, _ = dominant_color(hsv, pm)
        counts[name] = counts.get(name, 0) + 1
        parts.append((name, cv2.boundingRect(c)))
    reasons = []
    for k, exp in BOM.items():
        if counts[k] < exp:
            reasons.append(f"MISSING {exp - counts[k]} x {BOM_NAMES[k]}")
        elif counts[k] > exp:
            reasons.append(f"EXTRA {counts[k] - exp} x {BOM_NAMES[k]}")
    if counts["UNKNOWN"]:
        reasons.append(f"FOREIGN PART x {counts['UNKNOWN']}")
    if sum(counts.values()) == 0:
        return None, ["NO PARTS IN TRAY"], {"parts": [], "counts": counts}
    return len(reasons) == 0, reasons, {"parts": parts, "counts": counts}

# ---- Station 2: final assembly check
def order_box(box):
    pts = box[np.argsort(box[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.float32([top[0], top[1], bot[1], bot[0]])

def find_base(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    om = object_mask(hsv)
    om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if not touches_border(c, om.shape)]
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * img.shape[0] * img.shape[1]:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    src = order_box(box)
    # make the long side vertical
    if np.linalg.norm(src[1] - src[0]) > np.linalg.norm(src[3] - src[0]):
        src = np.float32([src[1], src[2], src[3], src[0]])
    return src

def grid_check(warp_hsv, W, H):
    r = int(CELL_R * W)
    cells, score = [], 0
    for ri, row in enumerate(EXPECTED):
        for ci, exp in enumerate(row.split()):
            x, y = int(GRID_X[ci] * W), int(GRID_Y[ri] * H)
            m = np.zeros(warp_hsv.shape[:2], np.uint8)
            cv2.circle(m, (x, y), r, 255, -1)
            found, _ = dominant_color(warp_hsv, m)
            want = {"R": ["RED"], "O": ["ORANGE"], "C": ["BLACK", "LIME"]}[exp]
            ok = found in want
            score += ok
            cells.append((ri, ci, x, y, exp, found, ok))
    # lime pin at the center of the 2x2 black brick
    lx = int((GRID_X[1] + GRID_X[2]) / 2 * W)
    ly = int((GRID_Y[1] + GRID_Y[2]) / 2 * H)
    m = np.zeros(warp_hsv.shape[:2], np.uint8)
    cv2.circle(m, (lx, ly), r, 255, -1)
    cv2.circle(m, (lx, ly), r // 3, 0, -1)          # ignore the hole
    lime_found, _ = dominant_color(warp_hsv, m)
    lime_ok = lime_found == "LIME"
    return cells, (lx, ly, lime_ok), score + lime_ok

CORNER = {(0, 0): "TOP-LEFT", (0, 3): "TOP-RIGHT", (3, 0): "BOTTOM-LEFT", (3, 3): "BOTTOM-RIGHT"}

def inspect_assembly(img):
    src = find_base(img)
    if src is None:
        return None, ["NO PART"], {}
    W, H = 300, 450
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    best = None
    for rot in (0, 2):                      # part may be rotated 180 degrees
        s = np.roll(src, -rot, axis=0)
        M = cv2.getPerspectiveTransform(s, dst)
        warp = cv2.warpPerspective(img, M, (W, H))
        hsv = cv2.cvtColor(cv2.GaussianBlur(warp, (5, 5), 0), cv2.COLOR_BGR2HSV)
        cells, lime, score = grid_check(hsv, W, H)
        if best is None or score > best[0]:
            best = (score, M, cells, lime, warp)
    _, M, cells, lime, warp = best
    reasons = []
    center_bad = False
    for ri, ci, x, y, exp, found, ok in cells:
        if ok:
            continue
        if exp == "R":
            reasons.append(f"RED MISSING at {CORNER[(ri, ci)]}")
        elif exp == "O" and found == "RED":
            reasons.append(f"RED MISPLACED at R{ri+1}C{ci+1}")
        elif exp == "O" and found in ("BLACK", "LIME"):
            center_bad = True
        elif exp == "C":
            center_bad = True
        else:
            reasons.append(f"UNEXPECTED {found} at R{ri+1}C{ci+1}")
    if center_bad:
        reasons.insert(0, "BLACK BRICK MISSING / SHIFTED")
    if not lime[2]:
        reasons.append("LIME PIN MISSING")
    return len(reasons) == 0, reasons, {"M": M, "cells": cells, "lime": lime, "warp": warp}

# ---------------------------------------------------------------- DRAWING
def put(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

def draw_kit(view, info, k):
    for name, (x, y, w, h) in info.get("parts", []):
        x, y, w, h = [int(v * k) for v in (x, y, w, h)]
        cv2.rectangle(view, (x, y), (x + w, y + h), DRAW[name], 3)
        put(view, name, (x, y - 8), 0.6, DRAW[name], 1)

def draw_assembly(view, info, k, show_grid):
    if "M" not in info:
        return
    Minv = np.linalg.inv(info["M"])
    def to_img(x, y):
        p = Minv @ np.array([x, y, 1.0])
        return int(p[0] / p[2] * k), int(p[1] / p[2] * k)
    W, H = 300, 450
    corners = [to_img(*p) for p in [(0, 0), (W, 0), (W, H), (0, H)]]
    cv2.polylines(view, [np.int32(corners)], True, (255, 255, 255), 2)
    if not show_grid:
        return
    r = max(6, int(CELL_R * W * k * np.linalg.norm(np.subtract(corners[1], corners[0])) / (W * k)))
    for ri, ci, x, y, exp, found, ok in info["cells"]:
        c = (0, 220, 0) if ok else (0, 0, 255)
        cv2.circle(view, to_img(x, y), r, c, 3 if ok else 5)
    lx, ly, lok = info["lime"]
    cv2.drawMarker(view, to_img(lx, ly), (0, 220, 0) if lok else (0, 0, 255),
                   cv2.MARKER_CROSS, r * 2, 4)

def draw_hud(view, station, result, reasons, ms, stats, auto):
    h, w = view.shape[:2]
    bar = view.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(bar, 0.75, view, 0.25, 0, view)
    title = "STATION 1 - KIT CHECK" if station == 1 else "STATION 2 - FINAL ASSEMBLY"
    put(view, title, (20, 46), 1.0, (255, 255, 255))
    put(view, f"{ms:5.1f} ms", (w - 190, 46), 0.9, (200, 200, 200))
    mode = "AUTO" if auto else "TRIGGER [SPACE]"
    put(view, mode, (w - 470, 46), 0.7, (0, 200, 255) if auto else (180, 180, 180), 1)
    if result is None:
        txt, col = "WAITING", (200, 200, 200)
    elif result:
        txt, col = "OK", (0, 210, 0)
    else:
        txt, col = "NG", (0, 0, 255)
    if result is not None:
        cv2.rectangle(view, (0, 70), (w, 78), col, -1)
    put(view, txt, (20, 170), 3.2 if txt != "WAITING" else 1.6, col, 6 if txt != "WAITING" else 3)
    y = 215
    for rs in reasons[:6]:
        put(view, rs, (25, y), 0.75, (0, 0, 255) if result is False else (220, 220, 220), 2)
        y += 34
    ok, ng = stats
    tot = ok + ng
    yld = f"{100 * ok / tot:.1f}%" if tot else "-"
    put(view, f"OK {ok}   NG {ng}   YIELD {yld}", (20, h - 25), 0.8, (255, 255, 255))
    put(view, VERSION, (w - 420, h - 25), 0.6, (160, 160, 160), 1)

# ---------------------------------------------------------------- MAIN
def roi_rect(img, roi):
    """roi = (size, dx, dy) -> pixel rectangle x0, y0, x1, y1 (clamped to the image)."""
    H, W = img.shape[:2]
    size, dx, dy = roi
    dx, dy = max(-0.5, min(0.5, dx)), max(-0.5, min(0.5, dy))
    side = int(min(H, W) * size)
    cx, cy = int(W / 2 + dx * W), int(H / 2 + dy * H)
    x0 = max(0, min(W - side, cx - side // 2))
    y0 = max(0, min(H - side, cy - side // 2))
    return x0, y0, x0 + side, y0 + side

def inspect(img, station, roi):
    x0, y0, x1, y1 = roi_rect(img, roi)
    crop = img[y0:y1, x0:x1]
    k = PROC_WIDTH / crop.shape[1]
    small = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    res = inspect_kit(small) if station == 1 else inspect_assembly(small)
    ms = (time.perf_counter() - t0) * 1000
    return res, ms, 1 / k

def render(img, station, res, ms, stats, auto, show_grid, show_mask, roi):
    result, reasons, info = res
    x0, y0, x1, y1 = roi_rect(img, roi)
    view = (img * 0.35).astype(np.uint8)               # dim everything outside the box
    crop = img[y0:y1, x0:x1]
    if show_mask:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        inner = np.zeros_like(crop)
        for name in COLORS:
            inner[color_mask(hsv, name) > 0] = DRAW[name]
    else:
        inner = crop.copy()
    view[y0:y1, x0:x1] = inner
    sub = view[y0:y1, x0:x1]                           # drawing on sub draws into view
    if station == 1:
        draw_kit(sub, info, scale_k(crop))
    else:
        draw_assembly(sub, info, scale_k(crop), show_grid)
    col = (200, 200, 200) if result is None else ((0, 210, 0) if result else (0, 0, 255))
    cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), col, 3)
    put(view, f"INSPECTION AREA  {int(roi[0] * 100)}%  [UP/DOWN]", (x0 + 8, y1 - 12), 0.6, col, 1)
    draw_hud(view, station, result, reasons, ms, stats, auto)
    return view

def scale_k(img):
    return img.shape[1] / PROC_WIDTH

def log_result(station, result, reasons, ms, frame, view):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    tag = "OK" if result else "NG"
    cv2.imwrite(os.path.join(LOG_DIR, f"{ts}_ST{station}_{tag}.jpg"), view)
    path = os.path.join(LOG_DIR, "log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["timestamp", "station", "result", "cycle_ms", "reasons"])
        wr.writerow([ts, station, tag, f"{ms:.1f}", " | ".join(reasons)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", help="test on an image instead of camera")
    ap.add_argument("--station", type=int, default=1)
    ap.add_argument("--out", help="with --image: save result image and exit")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    ap.add_argument("--roi", type=float, default=ROI_SIZE, help="inspection box size (0.1-1.0 of image height)")
    a = ap.parse_args()
    roi = [min(1.0, max(0.1, a.roi)), 0.0, 0.0]
    station, auto, show_grid, show_mask = a.station, True, True, False
    stats = [0, 0]

    if a.image:
        img = cv2.imread(a.image)
        res, ms, _ = inspect(img, station, roi)
        print(f"Station {station}: {'OK' if res[0] else 'NG' if res[0] is False else '-'}  {ms:.1f} ms  {res[1]}")
        view = render(img, station, res, ms, stats, auto, show_grid, show_mask, roi)
        if a.out:
            cv2.imwrite(a.out, view)
            return
        cap = None
    else:
        cap = cv2.VideoCapture(a.camera, cv2.CAP_AVFOUNDATION) if os.uname().sysname == "Darwin" \
            else cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            raise SystemExit("Camera not found. Try --camera 1 (and allow camera access for Terminal).")

    print(f"ChinarSoft LEGO Vision {VERSION}  -  click the video window before using the keys")
    WIN = "ChinarSoft Vision"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    full = False
    win_w = a.width
    hold = None                                  # frozen result after trigger
    first_frame = True
    while True:
        if cap is not None:
            ok, img = cap.read()
            if not ok:
                break
        trig = False
        raw = cv2.waitKeyEx(1)
        key = raw & 0xFF if 0 <= raw < 256 else raw
        if key in (ord("q"), 27):
            break
        elif key == ord("1"): station, hold = 1, None
        elif key == ord("2"): station, hold = 2, None
        elif key == ord("a"): auto, hold = not auto, None
        elif key == ord("g"): show_grid = not show_grid
        elif key == ord("m"): show_mask = not show_mask
        elif key == ord("r"): stats = [0, 0]
        elif key == ord(" "): trig = True
        elif key in (ord("+"), ord("=")) + KEY_UP: roi[0], hold = min(1.0, roi[0] + ROI_STEP), None
        elif key in (ord("-"), ord("_")) + KEY_DOWN: roi[0], hold = max(0.1, roi[0] - ROI_STEP), None
        elif key == ord("i"): roi[2], hold = max(-0.5, roi[2] - ROI_MOVE), None
        elif key == ord("k"): roi[2], hold = min(0.5, roi[2] + ROI_MOVE), None
        elif key == ord("j"): roi[1], hold = max(-0.5, roi[1] - ROI_MOVE), None
        elif key == ord("l"): roi[1], hold = min(0.5, roi[1] + ROI_MOVE), None
        elif key == ord("f"):
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if not full:
                first_frame = True
        elif key == ord("c"): roi[1], roi[2], hold = 0.0, 0.0, None
        elif raw != -1:
            print(f"Unknown key code: {raw}  (send this number to Claude)")

        if auto or trig:
            res, ms, _ = inspect(img, station, roi)
            view = render(img, station, res, ms, stats, auto, show_grid, show_mask, roi)
            if trig and res[0] is not None:
                stats[0 if res[0] else 1] += 1
                view = render(img, station, res, ms, stats, auto, show_grid, show_mask, roi)
                log_result(station, res[0], res[1], ms, img, view)
                if not auto:
                    hold = view
        if not auto:
            if hold is not None:
                view = hold
            else:
                view = render(img, station, (None, ["Press SPACE to inspect"], {}), 0, stats,
                              auto, show_grid, show_mask, roi)
        if not full:
            h, w = view.shape[:2]
            if w > win_w:                                  # shrink the picture to the window size
                view = cv2.resize(view, (win_w, int(h * win_w / w)), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, view)
        if first_frame:
            cv2.resizeWindow(WIN, view.shape[1], view.shape[0])
            first_frame = False

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
