import torch
import supervision as sv
import numpy as np

from logic.hotkeys_watcher import hotkeys_watcher
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse
from logic.shooting import shooting
from logic.model_classes import HEAD_CLASS_ID, is_head_class
from logic.team_filter import is_friendly

class Target:
    def __init__(self, x, y, w, h, cls):
        self.cls = int(cls)
        self.x = x
        self.y = y if is_head_class(self.cls) else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h

class FrameParser:
    def __init__(self):
        self.arch = self.get_arch()
        self._center = None
        self._center_cache_key = None
        self.locked_target_pos = None
        self.locked_frames_lost = 0
        self.current_frame = None   # run.py se set hota hai (team filter ke liye)

    def reset_lock(self):
        self.locked_target_pos = None
        self.locked_frames_lost = 0
        try:
            mouse.reset_filter()
        except Exception:
            pass

    def parse(self, result):
        if result is None:
            self._handle_no_detections()
            return

        if isinstance(result, sv.Detections):
            self._process_sv_detections(result)
        else:
            self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        if len(detections) > 0:
            if cfg.show_window or cfg.show_overlay:
                visuals.draw_helpers(detections)
            target = self.sort_targets(detections)
            self._handle_target(target)
        else:
            self._handle_no_detections()

    def _process_yolo_detections(self, results):
        frames = (results,) if hasattr(results, "boxes") else results
        processed = False

        for frame in frames:
            processed = True
            boxes = getattr(frame, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                target = self.sort_targets(frame)
                self._handle_target(target)
                self._visualize_frame(frame)
            else:
                self._handle_no_detections()

        if not processed:
            self._handle_no_detections()

    def _handle_target(self, target):
        if target:
            # Agar F1 se shooting disabled hai to mouse/shoot mat karo, detection chalti rahegi
            if not hotkeys_watcher.auto_mode_enabled:
                shooting.shoot(False, False)
                return

            if hotkeys_watcher.clss is None:
                hotkeys_watcher.active_classes()

            if target.cls in hotkeys_watcher.clss:
                # Team filter: apni team par fire nahi karni
                if is_friendly(self.current_frame, target.x, target.y, target.w, target.h, target.cls):
                    self._handle_no_detections()
                    return
                mouse.process_data((target.x, target.y, target.w, target.h, target.cls))

    def _visualize_frame(self, frame):
        if cfg.show_window or cfg.show_overlay:
            if cfg.show_boxes or cfg.overlay_show_boxes:
                visuals.draw_helpers(frame.boxes)

            if cfg.show_window and cfg.show_detection_speed:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])

    def _handle_no_detections(self):
        self.locked_frames_lost += 1
        # Target memory: Target ko 15-25 frames tak pakad kar rakhe, chhore na
        max_lost = 25 if shooting.button_pressed else 15
        if self.locked_frames_lost > max_lost:
            self.locked_target_pos = None
            try:
                mouse.reset_filter()
            except Exception:
                pass
        if cfg.show_window or cfg.show_overlay:
            visuals.clear()
        if cfg.auto_shoot or cfg.triggerbot:
            shooting.shoot(False, False)

    def sort_targets(self, frame):
        if isinstance(frame, sv.Detections):
            boxes_array, classes_tensor = self._convert_sv_to_tensor(frame)
        else:
            if frame.boxes is None or len(frame.boxes) == 0:
                return None
            boxes_array = self._to_tensor(frame.boxes.xywh, dtype=torch.float32)
            classes_tensor = self._to_tensor(frame.boxes.cls, dtype=torch.long)

        if not classes_tensor.numel():
            return None

        return self._find_nearest_target(boxes_array, classes_tensor)

    def _convert_sv_to_tensor(self, frame):
        xyxy = np.asarray(frame.xyxy, dtype=np.float32)
        xywh = np.empty((xyxy.shape[0], 4), dtype=np.float32)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        boxes_tensor = torch.as_tensor(xywh, dtype=torch.float32, device=self.arch)
        class_ids = frame.class_id if frame.class_id is not None else np.zeros(xyxy.shape[0], dtype=np.int64)
        classes_tensor = torch.as_tensor(class_ids, dtype=torch.long, device=self.arch)
        return boxes_tensor, classes_tensor

    def _to_tensor(self, value, dtype):
        if hasattr(value, "to"):
            return value.to(device=self.arch, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.arch)

    def _find_nearest_target(self, boxes_array, classes_tensor):
        candidate_idxs = self._get_active_candidate_idxs(classes_tensor)

        if candidate_idxs.numel() == 0:
            self.locked_target_pos = None
            return None

        head_candidate_mask = classes_tensor[candidate_idxs] == HEAD_CLASS_ID

        if cfg.disable_headshot:
            non_head_candidate_idxs = candidate_idxs[~head_candidate_mask]
            if non_head_candidate_idxs.numel() == 0:
                self.locked_target_pos = None
                return None
            candidate_idxs = non_head_candidate_idxs
        else:
            if head_candidate_mask.any():
                candidate_idxs = candidate_idxs[head_candidate_mask]

        # TEAM FILTER: Apni team ke members ko filter out karo taake sirf ENEMIES par aim ho
        if cfg.team_filter_enabled and self.current_frame is not None and candidate_idxs.numel() > 0:
            enemy_candidates = []
            for c_idx in candidate_idxs:
                bx, by, bw, bh = boxes_array[c_idx, :4]
                bcls = int(classes_tensor[c_idx].item())
                if not is_friendly(self.current_frame, float(bx), float(by), float(bw), float(bh), bcls):
                    enemy_candidates.append(c_idx.item())

            if len(enemy_candidates) > 0:
                candidate_idxs = torch.tensor(enemy_candidates, dtype=torch.long, device=boxes_array.device)
            else:
                # Sab samne wale teammates hain! Inpar lock/fire mat karo
                self.locked_target_pos = None
                return None

        # STICKY TARGET LOCK: Follow the same enemy as they move
        chosen_idx = None
        if self.locked_target_pos is not None:
            last_x, last_y = self.locked_target_pos
            ref_pos = torch.tensor([last_x, last_y], dtype=torch.float32, device=boxes_array.device)
            dist_to_locked = torch.sum((boxes_array[candidate_idxs, :2] - ref_pos) ** 2, dim=1)
            min_dist_val, min_dist_sub = torch.min(dist_to_locked, dim=0)

            # Follow radius (up to 350px so fast running enemies are never dropped)
            if min_dist_val.item() <= (350.0 ** 2):
                chosen_idx = candidate_idxs[min_dist_sub].item()
                self.locked_frames_lost = 0
            else:
                self.locked_frames_lost += 1
                max_lost_frames = 12 if shooting.button_pressed else 6
                if self.locked_frames_lost > max_lost_frames:
                    self.locked_target_pos = None

        # If not tracking or target lost, pick target closest to screen crosshair center
        if chosen_idx is None:
            center = self._get_center(boxes_array.device)
            distances_sq = torch.sum((boxes_array[candidate_idxs, :2] - center) ** 2, dim=1)
            min_center_sub = torch.argmin(distances_sq).item()
            chosen_idx = candidate_idxs[min_center_sub].item()
            self.locked_frames_lost = 0

        target_data = boxes_array[chosen_idx, :4].cpu().numpy()
        target_class = int(classes_tensor[chosen_idx].item())
        target = Target(*target_data, target_class)

        # Update locked position for continuous tracking
        self.locked_target_pos = (target.x, target.y)

        return target

    def _get_active_candidate_idxs(self, classes_tensor):
        if hotkeys_watcher.clss is None:
            hotkeys_watcher.active_classes()

        active_class_ids = hotkeys_watcher.clss or []
        if not active_class_ids:
            return torch.empty(0, dtype=torch.long, device=classes_tensor.device)

        active_mask = torch.zeros_like(classes_tensor, dtype=torch.bool)
        for class_id in active_class_ids:
            active_mask |= classes_tensor == int(class_id)

        return torch.nonzero(active_mask, as_tuple=False).flatten()

    def _get_center(self, device):
        key = (capture.screen_x_center, capture.screen_y_center, str(device))
        if self._center is None or self._center_cache_key != key:
            self._center = torch.tensor(
                [capture.screen_x_center, capture.screen_y_center],
                dtype=torch.float32,
                device=device
            )
            self._center_cache_key = key
        return self._center

    def update_settings(self):
        self.arch = self.get_arch()
        self._center = None
        self._center_cache_key = None

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        ai_device = str(cfg.AI_device).lower()
        if 'cpu' in ai_device:
            return 'cpu'
        if ai_device == 'cuda':
            return 'cuda'
        if ai_device.startswith('cuda:'):
            return cfg.AI_device
        else:
            return f'cuda:{cfg.AI_device}'

frameParser = FrameParser()
