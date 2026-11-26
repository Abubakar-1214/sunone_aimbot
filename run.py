from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from ultralytics import YOLO
import torch
import supervision as sv
from trackers import ByteTrackTracker

from logic.config_watcher import cfg
from logic.checks import run_checks
from logic.logger import logger

@torch.inference_mode()
def perform_detection(model, image, tracker: ByteTrackTracker | None = None):
    ai_device = str(cfg.AI_device).lower()
    kwargs = dict(
        source=image,
        imgsz=cfg.ai_model_image_size,
        conf=cfg.AI_conf,
        iou=0.50,
        device=cfg.AI_device,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=1,
        visualize=False,
        verbose=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False,
        stream=True
    )

    if "cpu" not in ai_device:
        kwargs["half"] = True

    kwargs["cfg"] = "logic/tracker.yaml" if tracker else "logic/game.yaml"

    results = model.predict(**kwargs)

    if tracker:
        for res in results:
            det = sv.Detections.from_ultralytics(res)
            return tracker.update(det)
    else:
        return next(results, None)

def ask_team_selection():
    target_team = "TERRORIST (T)" if cfg.my_team.upper() == "CT" else "COUNTER-TERRORIST (CT)"
    filter_status = "ENABLED [F6]" if cfg.team_filter_enabled else "DISABLED [F6] (Deathmatch Mode)"
    print("\n" + "=" * 56)
    print("        CS2 ACTIVE AIMBOT RUNNING")
    print("=" * 56)
    print(f"[AIMBOT] AAPKI TEAM:   [{cfg.my_team.upper()}] (Press [F5] to Switch)")
    print(f"[AIMBOT] TARGET ENEMY:  [{target_team}]")
    print(f"[AIMBOT] TEAM FILTER:   {filter_status}")
    print("=" * 56 + "\n")

def init():
    run_checks()
    ask_team_selection()
    from logic.capture import capture
    from logic.visual import visuals
    from logic.frame_parser import frameParser
    from logic.hotkeys_watcher import hotkeys_watcher
    from logic.shooting import shooting

    tracker = ByteTrackTracker() if not cfg.disable_tracker else None
    
    try:
        model = YOLO(f"models/{cfg.AI_model_name}", task="detect")
    except Exception as e:
        logger.error(f"An error occurred when loading the AI model:\n{e}")
        raise SystemExit(0)
        
    while True:
        image = capture.get_new_frame()

        if image is None:
            continue

        if cfg.circle_capture:
            image = capture.convert_to_circle(image)

        if hotkeys_watcher.app_pause != 0:
            visuals.clear()
            shooting.shoot(False, False)
            if cfg.show_window or cfg.show_overlay:
                visuals.submit_frame(image)
            continue

        result = perform_detection(model, image, tracker)
        frameParser.current_frame = image   # team_filter ke liye frame pass karna
        frameParser.parse(result)

        if cfg.show_window or cfg.show_overlay:
            visuals.submit_frame(image)

if __name__ == "__main__":
    init()
