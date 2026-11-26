"""
Team Filter — CS2 Team & Enemy Recognition
CT Uniform: Dark Navy, Slate, Cool Grey, Black (Cold tones)
T Uniform:  Khaki, Tan, Sand, Brown, Orange, Olive (Warm earth tones)
"""
import cv2
import numpy as np
from logic.config_watcher import cfg


def get_torso_crop(frame, cx: float, cy: float, w: float, h: float, cls: int):
    if frame is None:
        return None

    fh, fw = frame.shape[:2]

    if cls == 1:  # HEAD
        torso_cy = cy + h * 2.0
        torso_cx = cx
        torso_w = max(w * 1.6, 14.0)
        torso_h = max(h * 1.8, 16.0)
    else:  # Body
        torso_cy = cy - h * 0.15
        torso_cx = cx
        torso_w = max(w * 0.50, 14.0)
        torso_h = max(h * 0.35, 16.0)

    x1 = max(0, int(torso_cx - torso_w / 2))
    y1 = max(0, int(torso_cy - torso_h / 2))
    x2 = min(fw - 1, int(torso_cx + torso_w / 2))
    y2 = min(fh - 1, int(torso_cy + torso_h / 2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def classify_crop(crop_bgr) -> str:
    if crop_bgr is None or crop_bgr.size < 16:
        return "UNKNOWN"

    flat_bgr = crop_bgr.reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    flat_hsv = hsv.reshape(-1, 3).astype(np.float32)

    med_b, med_g, med_r = np.median(flat_bgr, axis=0)
    med_h, med_s, med_v = np.median(flat_hsv, axis=0)

    ct_score = 0
    t_score = 0

    # T side: strong warm earth tones (Red & Green significantly higher than Blue)
    if (med_r > med_b + 14) and (med_g > med_b + 5):
        t_score += 4
    elif (med_r > med_b + 8):
        t_score += 2

    if (10 <= med_h <= 55) and med_s >= 25:
        t_score += 3

    # CT side: cold navy / slate / cool grey
    diff_br = med_b - med_r
    if diff_br >= 3:
        ct_score += 4
    elif diff_br >= -3 and med_s < 45:
        ct_score += 3

    if (75 <= med_h <= 145) and med_s >= 15:
        ct_score += 4

    if ct_score >= 3 and t_score <= 1:
        return "CT"
    elif t_score >= 3 and ct_score <= 1:
        return "T"
    elif ct_score > t_score:
        return "CT"
    elif t_score > ct_score:
        return "T"

    return "UNKNOWN"


def get_target_team(frame, cx: float, cy: float, w: float, h: float, cls: int = 1) -> str:
    crop = get_torso_crop(frame, cx, cy, w, h, cls)
    if crop is None or crop.size == 0:
        return "UNKNOWN"
    return classify_crop(crop)


def is_friendly(frame, cx: float, cy: float, w: float, h: float, cls: int = 1) -> bool:
    """
    True  = Teammate hai (SKIP - FIRE MAT KARO!)
    False = Enemy hai (AIM AUR FIRE KARO!)
    """
    if not getattr(cfg, 'team_filter_enabled', False):
        return False  # Filter disabled: target everyone!

    if frame is None:
        return False

    detected_team = get_target_team(frame, cx, cy, w, h, cls)
    my_team = str(getattr(cfg, 'my_team', 'CT')).upper().strip()

    # Sirf tab friendly manenge agar TARGET CONFIRMED AAPKI TEAM KA HO:
    if detected_team == my_team:
        return True

    # Baki sab cases mein (chahe dushman ho ya UNKNOWN) -> Target karo, miss mat karo!
    return False
