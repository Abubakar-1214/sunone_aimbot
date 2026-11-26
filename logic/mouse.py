import time
import math
import os
import threading
import numpy as np
import supervision as sv

from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.logger import logger
from logic.platform import input_backend
from logic.model_classes import is_head_class, HEAD_CLASS_ID

if cfg.mouse_rzr:
    from logic.rzctl import RZCONTROL

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("[Mouse] filterpy not available - Kalman prediction disabled. pip install filterpy")


class KalmanTargetTracker:
    """
    Kalman Filter for 2D position + velocity tracking.
    State: [x, y, vx, vy] — tracks WHERE the target is and HOW FAST it's moving.
    Predicts future position so the mouse can lead moving targets.
    """
    def __init__(self, init_x, init_y):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        self.kf.P *= 1000.0
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=3.0, block_size=2, order_by_dim=False)
        self.kf.R = np.array([
            [1.0, 0],
            [0, 1.0]], dtype=np.float32)

        self.kf.x = np.array([init_x, init_y, 0, 0], dtype=np.float32)
        self.last_update_time = time.perf_counter()
        self.initialized = True
        self.update_count = 0

    def update(self, x, y):
        now = time.perf_counter()
        dt = now - self.last_update_time
        self.last_update_time = now
        self.update_count += 1

        if dt > 0.001:
            self.kf.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

        self.kf.update(np.array([x, y], dtype=np.float32))
        self.kf.predict()

    def get_velocity(self):
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.x[2]), float(self.kf.x[3])

    def get_speed(self):
        vx, vy = self.get_velocity()
        return math.hypot(vx, vy)

    def get_position(self):
        return float(self.kf.x[0]), float(self.kf.x[1])

    def get_predicted_position(self, lookahead_seconds=0.03):
        if not self.initialized:
            return float(self.kf.x[0]), float(self.kf.x[1])
        px = float(self.kf.x[0]) + float(self.kf.x[2]) * lookahead_seconds
        py = float(self.kf.x[1]) + float(self.kf.x[3]) * lookahead_seconds
        return px, py

    def reset(self, x=None, y=None):
        if x is not None and y is not None:
            self.kf.x = np.array([x, y, 0, 0], dtype=np.float32)
        self.kf.P *= 1000.0
        self.last_update_time = time.perf_counter()
        self.update_count = 0


class OneEuroFilter:
    """
    1-Euro Filter for jitter reduction.
    Low speed: heavy smoothing. High speed: minimal smoothing.
    """
    def __init__(self, x0, min_cutoff=1.5, beta=0.035, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = time.perf_counter()

    def filter(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        a_d = (2 * math.pi * self.d_cutoff * dt) / (2 * math.pi * self.d_cutoff * dt + 1)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = (2 * math.pi * cutoff * dt) / (2 * math.pi * cutoff * dt + 1)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class MouseThread:
    """
    Aim control: "foran detected point par jao, aur wahi lock ho jao".

    WHY THIS IS RELATIVE-ONLY
    -------------------------
    An FPS reads mouse input through Raw Input (WM_INPUT / DirectInput), NOT
    through the OS cursor. So SetCursorPos / MOUSEEVENTF_ABSOLUTE move the
    *desktop cursor* only and the game view does not follow at all. Games only
    react to *relative* deltas (MOUSEEVENTF_MOVE), which is what we send.

    WHY OLD BUILDS COULD NOT LOCK
    -----------------------------
      * the whole move was computed on the detection frame only (34 FPS on
        this machine = one chance every 29 ms), so between two detections the
        crosshair just sat there -> looked like "mouse foran nahi ata",
      * one open-loop guess per frame, sized with an ESTIMATED px-per-count,
        always left a residual that produced another step next frame -> the
        crosshair glided past the point, and nothing ever said "arrived, stop".

    HOW THIS BUILD DOES IT
    ----------------------
    1. FEEDBACK-FREE RESPONSE, NO BAKED-IN CURVE: the target point is assumed
       static for the short time it takes to reach it. The first step is the
       whole remaining offset, corrected later by what the game really did.
       There is no exponential "gain ramp" that needed 5+ detection frames.
    2. HIGH-RATE AIM LOOP (`aim_loop`, ~250 Hz): moves are NOT limited to the
       detection frame rate. The loop corrects every `mouse_aim_tick_ms`
       against the last measured goal, so arrival happens in a few ms instead
       of "next detection".
    3. DEPTH-SCALED ROTATION MODEL: a head twice as far away needs half the
       rotation for the same pixel offset. The 2D box height IS that distance
       signal, so the px-per-count of THIS target is derived from its box size
       (`depth_fov_deg`, `depth_ref_height_px`) instead of a fixed number that
       is only right for one distance.
    4. LIVE CALIBRATION: every step remembers what it sent; the measured
       change of the offset tells the real px-per-count. A global scale
       (`_aim_scale`) converges to the game/DPI/sens truth, and the per-target
       depth ratio is corrected by `_aim_depth_scale` — so both "dur" and
       "nazdeek" targets get the right amount of movement.
    5. ARRIVE AND FREEZE: inside `mouse_deadzone` nothing is sent, so the
       crosshair physically cannot slide past the head.
    6. RELEASE ON DETECTION LOSS: the aim goal has a hard TTL
       (`mouse_release_ms`, default 0 = release on the very next frame).
       When the model stops detecting, `release_target()` clears the goal and
       the mouse is free immediately.
    """

    def __init__(self):
        self._io_lock = threading.Lock()
        self._goal = None                 # (x, y, w, h, cls, stamp, id)
        self._goal_id = 0
        self._seen_goal_id = 0
        self._hw_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._aim_thread = None
        self._aim_wake = threading.Event()
        self.error_px = 999.0             # last measured distance to the goal
        self.aim_log = []                 # (t, counts_x, counts_y, err_px)

        self.initialize_parameters()
        self.setup_hardware()

    @property
    def _seed_pxpc(self):
        """Seed px-per-count, in the same space as `_world_scale`."""
        return max(1e-6, 0.5 * (self.px_per_count_x + self.px_per_count_y))

    # ------------------------------------------------------- parameters
    def initialize_parameters(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.speed_correction_factor = 0.1
        self.bScope = False
        self.arch = self.get_arch()
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100

        self.windows_sens = getattr(cfg, 'windows_sensitivity', 6)
        self.m_yaw = getattr(cfg, 'm_yaw', 0.022)
        self.m_pitch = getattr(cfg, 'm_pitch', 0.022)
        self.deadzone_px = getattr(cfg, 'mouse_deadzone', 4.0)
        self.max_move_per_frame = getattr(cfg, 'max_move_per_frame', 500)

        # 1-Euro filter state
        self.filter_x = None
        self.filter_y = None

        # Kalman tracker for velocity prediction
        self.kalman_tracker = None
        self.kalman_enabled = KALMAN_AVAILABLE and not cfg.disable_prediction
        self.kalman_lookahead = float(getattr(cfg, 'prediction_interval', 0.045))

        # Snap-jump config
        self.snap_jump_enabled = getattr(cfg, 'head_snap_mode', True)
        self.snap_factor = float(getattr(cfg, 'head_snap_factor', 1.0))
        self.first_target_seen = False

        # Lock state
        self.locked_on_target = False
        self.hard_lock = False
        self.consecutive_lock_frames = 0

        # PID (optional, off by default)
        self.pid_enabled = getattr(cfg, 'pid_enabled', False)
        self.pid_kp = float(getattr(cfg, 'pid_kp', 0.65))
        self.pid_ki = float(getattr(cfg, 'pid_ki', 0.08))
        self.pid_kd = float(getattr(cfg, 'pid_kd', 0.0003))
        self.pid_x = None
        self.pid_y = None

        # === MAGNETIC LOCK (relative, closed loop) ===
        self.magnetic_strength = float(getattr(cfg, 'magnetic_strength', 4.0))
        self.magnetic_smoothing = float(getattr(cfg, 'magnetic_smoothing', 0.20))
        self.decay_radius = 100.0 / max(1.0, self.magnetic_strength)

        self._recalc_conversion_factor()
        self._init_lock_controller()

    def _init_lock_controller(self):
        """All aim-controller tuning lives here (reads `[Mouse]` config)."""
        # zone: beyond this the aim leaves the mouse alone (px, detection view)
        self.lock_max_zone_px = float(getattr(cfg, 'lock_max_zone_px', 4000.0))
        self.lock_max_move_px = float(getattr(cfg, 'lock_max_move_px', 140.0))
        self.lock_settle_px = float(getattr(cfg, 'lock_settle_px', 2.0))

        # arrival: the step may cover the whole remaining offset, once the
        # calibration is warm. While it is cold the uncertainty scales it down.
        self.lock_sens_guard = float(getattr(cfg, 'lock_sens_guard', 3.0))
        self.lock_precise_gain = float(getattr(cfg, 'lock_precise_gain', 1.0))

        # response timing
        self.reaction_delay = float(getattr(cfg, 'mouse_reaction_delay_ms', 0.0)) / 1000.0
        self.release_after = float(getattr(cfg, 'mouse_release_ms', 0.0)) / 1000.0
        self.aim_tick = max(0.001, float(getattr(cfg, 'mouse_aim_tick_ms', 4.0)) / 1000.0)

        # depth model (dur / nazdeek targets get the right amount of movement)
        # `depth_fov_deg` = the game's HORIZONTAL FOV for the sensitivity you
        # entered. Optional: while `depth_ref_height_px` is 0 the controller
        # learns the ratio from live head-box sizes.
        self.depth_fov_deg = float(getattr(cfg, 'depth_fov_deg', 0.0) or 0.0)
        self.depth_ref_height_px = float(getattr(cfg, 'depth_ref_height_px', 0.0) or 0.0)
        self.depth_mode = "box" if self.depth_ref_height_px > 1.0 else "auto"

        # prediction: only used to OFFSET the goal for a moving target
        self.predict_lookahead = float(getattr(cfg, 'predict_lookahead_ms', 0.0)) / 1000.0

        # calibration state
        self._aim_depth_scale = 1.0
        self._aim_scale_var = self.lock_sens_guard ** 2
        self._aim_samples = 0
        # "world scale" = true px-per-count expressed in the SEED's units.
        # Stored instead of a raw multiplier so re-reading config or changing
        # display/sens cannot silently rescale the calibration.
        # `_k_seed`: true px-per-seed-count, 1.0 = the config seed is exact.
        self._k_seed = float(getattr(cfg, 'mouse_calib_scale', 1.0)) or 1.0
        self._world_scale = self._k_seed
        self._world_depth = 1.0
        self._pending_counts = 0.0
        self._pending_vec = (0.0, 0.0)
        self._pending_err = None
        self._pending_err_x = None
        self._pending_err_y = None
        self._pending_sent = (0.0, 0.0)
        self._pending_line_x = None
        self._pending_line_y = None
        self._pending_h = 0.0
        self._pending_pxpc = None
        self._pending_new_goal = False
        self._step_gain = 1.0
        self._in_flight = False
        self._k_x = self._k_seed
        self._k_y = self._k_seed
        # axis direction: `_inverted` is the decision, `_flip_hits` counts
        # contradictions, `_flip_pending` marks a still unconfirmed guess
        self._inverted = [False, False]
        self._flip_hits = [0, 0]
        # how many real motion observations each axis has given, used to tell
        # "direction established" from "still probing" (see n_sign_known)
        self._axis_samples = [0, 0]
        self._cur_offset = None
        self._kal = None

        # Set by process_data() before calc_movement()
        self.human_moving = False

    def _reset_loop(self):
        """Forget per-target calibration + pending step."""
        self.hard_lock = False
        self._pending_counts = 0.0
        self._pending_vec = (0.0, 0.0)
        self._pending_err = None
        self._pending_err_x = None
        self._pending_err_y = None
        self._pending_sent = (0.0, 0.0)
        self._pending_line_x = None
        self._pending_line_y = None
        self._pending_h = 0.0
        self._pending_pxpc = None
        self._cur_offset = None
        self._aim_samples = 0
        self._kal = None
        self.human_moving = False

    def reset_filter(self):
        self.filter_x = None
        self.filter_y = None
        self.kalman_tracker = None
        self.first_target_seen = False
        self.locked_on_target = False
        self.consecutive_lock_frames = 0
        self._reset_loop()
        if hasattr(self, 'pid_x') and self.pid_x is not None:
            self.pid_x.reset()
        if hasattr(self, 'pid_y') and self.pid_y is not None:
            self.pid_y.reset()

    def reset_ema(self):
        self.reset_filter()

    def _recalc_conversion_factor(self):
        """Seed values only — the live calibration corrects them anyway."""
        sens = max(0.01, float(self.mouse_sensitivity))
        m_yaw = max(0.0001, float(self.m_yaw))
        m_pitch = max(0.0001, float(self.m_pitch))

        deg_per_count_x = sens * m_yaw
        deg_per_count_y = sens * m_pitch

        disp_w = float(getattr(cfg, 'display_width', 0))
        disp_h = float(getattr(cfg, 'display_height', 0))
        if disp_w <= 0 or disp_h <= 0:
            try:
                from screeninfo import get_monitors
                monitors = list(get_monitors())
                primary = next((m for m in monitors if m.is_primary), monitors[0]) if monitors else None
                disp_w = float(primary.width) if primary else 1920.0
                disp_h = float(primary.height) if primary else 1080.0
            except Exception:
                disp_w = 1920.0
                disp_h = 1080.0

        aspect = disp_w / max(1.0, disp_h)
        if aspect < 1.5:
            fov_x_eff = 90.0
            fov_y_eff = 73.74
        else:
            fov_x_eff = max(1.0, float(self.fov_x))
            fov_y_eff = max(1.0, float(self.fov_y))

        deg_per_px_x = fov_x_eff / disp_w
        deg_per_px_y = fov_y_eff / disp_h

        base_counts_x = deg_per_px_x / deg_per_count_x
        base_counts_y = deg_per_px_y / deg_per_count_y
        calib_scale = float(getattr(cfg, 'mouse_calib_scale', 1.0))

        self.counts_per_px_x = base_counts_x * calib_scale
        self.counts_per_px_y = base_counts_y * calib_scale
        self.px_per_count_x = 1.0 / max(1e-5, self.counts_per_px_x)
        self.px_per_count_y = 1.0 / max(1e-5, self.counts_per_px_y)
        self.display_w = disp_w
        self.display_h = disp_h

        logger.info(f"[Mouse] counts/px: x={self.counts_per_px_x:.4f} y={self.counts_per_px_y:.4f} | FOV: {fov_x_eff}x{fov_y_eff} | Disp: {disp_w}x{disp_h} | calib={calib_scale}")

    # ------------------------------------------------------------- depth
    def _depth_ratio(self, target_h):
        """
        How much MORE rotation this target needs than the seed calibration
        assumes (1.0 = exactly what the config seed predicted).

        Rotation for `px` is  px * FOV / (screen_px * count_scale * distance),
        so a target half as far needs twice the counts for the same pixel.
        The 2D box height is inversely proportional to distance.

          * depth_ref_height_px set -> ratio = ref / height  (pure geometry)
          * not set                -> ratio = auto_ref / height, where auto_ref
            is the running median head height and `_aim_depth_scale` carries
            the geometric correction learned from real arrivals.
        """
        h = float(target_h) if target_h else 0.0
        if h <= 1.0:
            return 1.0
        ref = float(getattr(self, 'depth_ref_height_px', 0.0) or 0.0)
        if ref <= 1.0:
            ref = float(getattr(self, 'depth_ref_height', 0.0) or 0.0)
        if ref <= 1.0:
            return 1.0
        return max(0.2, min(6.0, ref / h))

    def _update_depth_reference(self, target_h):
        """Track a running reference head height (slow EMA)."""
        h = float(target_h) if target_h else 0.0
        if h <= 1.0:
            return
        ref = getattr(self, 'depth_ref_height', 0.0)
        if ref <= 1.0:
            self.depth_ref_height = h
        else:
            self.depth_ref_height = 0.92 * ref + 0.08 * h

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
        return f'cuda:{cfg.AI_device}'

    def setup_hardware(self):
        if cfg.mouse_ghub and not hasattr(self, 'ghub'):
            from logic.ghub import gHub
            self.ghub = gHub

        if cfg.mouse_rzr and not hasattr(self, 'rzr'):
            from logic.rzctl import RZCONTROL
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rzctl.dll")
            self.rzr = RZCONTROL(dll_path)
            if not self.rzr.init():
                logger.error("Failed to initialize rzctl")

    # ------------------------------------------------------------ target
    def process_data(self, data):
        """Called by frame_parser for every detection (one per model frame)."""
        if isinstance(data, sv.Detections):
            xyxy = data.xyxy[0]
            target_x = (xyxy[0] + xyxy[2]) / 2
            target_y = (xyxy[1] + xyxy[3]) / 2
            target_w = xyxy[2] - xyxy[0]
            target_h = xyxy[3] - xyxy[1]
            target_cls = data.class_id[0] if data.class_id is not None and data.class_id.size > 0 else None
        else:
            target_x, target_y, target_w, target_h, target_cls = data

        shooting_state = self.get_shooting_key_state()

        self.visualize_target(target_x, target_y, target_cls)

        self.bScope = self.check_target_in_scope(
            target_x, target_y, target_w, target_h, self.bScope_multiplier
        ) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        # Hand the raw detection to the high-rate aim loop. The loop owns the
        # actual mouse movement, the per-frame code only decides WHERE to aim.
        self.set_target(target_x, target_y, target_w, target_h, target_cls)

        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        dist = math.hypot(offset_x, offset_y)

        self.visualize_history(target_x, target_y)
        shooting.submit(self.bScope, shooting_state)

        self.locked_on_target = self.hard_lock
        if self.hard_lock:
            self.consecutive_lock_frames += 1
        else:
            self.consecutive_lock_frames = 0

        logger.debug(
            f"[Mouse] target=({target_x:.0f},{target_y:.0f}) offset=({offset_x:+.0f},{offset_y:+.0f}) "
            f"dist={dist:.0f}px err={self.error_px:.1f}px "
            f"scale={self._aim_scale:.3f} depth={self._aim_depth_scale:.3f} locked={self.locked_on_target}"
        )

    def set_target(self, x, y, w=0.0, h=0.0, cls=None):
        """Publish a new aim goal (thread-safe). Adds the motion prediction."""
        gx, gy = float(x), float(y)
        if self.predict_lookahead > 0.0 and self._kal is not None:
            sx, sy = self._kal.predict(gx, gy)
            gx += sx * self.predict_lookahead
            gy += sy * self.predict_lookahead

        with self._io_lock:
            self._goal_id += 1
            self._goal = (gx, gy, float(w or 0.0), float(h or 0.0), cls,
                          time.perf_counter(), self._goal_id)
        # A brand-new goal must not wait for the next tick: wake the loop now,
        # that is what makes the crosshair leave instantly on a new detection.
        self._aim_wake.set()

    def release_target(self):
        """Detection gone: drop the goal, the mouse is free from now on."""
        with self._io_lock:
            self._goal = None
        self._reset_loop()
        self.hard_lock = False
        self.locked_on_target = False
        self.consecutive_lock_frames = 0

    def _current_goal(self):
        goal = self._goal
        if goal is None:
            return None
        if self.release_after > 0.0 and (time.perf_counter() - goal[5]) > self.release_after:
            return None
        return goal

    # -------------------------------------------------------- aim thread
    def start(self):
        if self._aim_thread is not None and self._aim_thread.is_alive():
            return
        self._stop_event.clear()
        self._aim_thread = threading.Thread(target=self.aim_loop, name="AimLoop", daemon=True)
        self._aim_thread.start()

    def stop(self):
        self._stop_event.set()
        self._aim_wake.set()

    def get_micro_move(self):
        """One aim correction against the current goal. (0,0) = locked/free."""
        goal = self._current_goal()
        if goal is None:
            self.hard_lock = False
            return 0.0, 0.0
        gx, gy, _w, h, _cls, _stamp, _gid = goal
        return self.calc_movement(gx, gy, target_h=h)

    def aim_loop(self):
        """
        High-rate aim loop.

        The detection thread only publishes a goal. This loop keeps correcting
        towards that goal as fast as `mouse_aim_tick_ms` allows (4 ms by
        default), so the crosshair arrives in milliseconds instead of waiting
        for the next model frame. The moment a goal is published the loop
        fires immediately (no tick wait on the first move).
        """
        tick = self.aim_tick
        next_tick = time.perf_counter()
        shooting_state = False
        kv_tick = 0
        while not self._stop_event.is_set():
            goal = self._current_goal()
            if goal is None:
                self.hard_lock = False
                # clear BEFORE waiting: a goal published during the wait must
                # keep the event set so this loop wakes on the spot
                self._aim_wake.clear()
                self._aim_wake.wait(0.05)
                next_tick = time.perf_counter()
                continue

            kv_tick += 1
            if kv_tick % 24 == 0:
                shooting_state = self.get_shooting_key_state()
            if kv_tick % 72 == 0:
                kv_tick = 0

            mx, my = self.get_micro_move()
            if mx or my:
                self.move_mouse(mx, my, shooting_state)

            now = time.perf_counter()
            next_tick += tick
            sleep_for = next_tick - now
            if sleep_for > 0.0:
                # a new goal published meanwhile interrupts the wait so the
                # first correction lands immediately instead of a tick later
                self._aim_wake.wait(sleep_for)
                self._aim_wake.clear()
            else:
                next_tick = now

    # ------------------------------------------------------------ control
    def _calib_uncertainty(self):
        """
        Relative uncertainty of the px-per-count scaling, 0.0 = known exactly.

        Grows while the calibration is cold (the config estimate can be a few
        times off) and shrinks as real observations arrive. Used to (a) soften
        the very first step and (b) bound the residual after a step so a wrong
        seed can never fly past the head.
        """
        return min(float(self.lock_sens_guard), max(0.0, math.sqrt(max(0.0, self._aim_scale_var))))

    # -- calibration math ------------------------------------------------
    # Model, per axis, in true pixels:
    #     err_after = err_before - counts * true_px_per_count
    # so `counts` (what we sent since the previous detection) against
    # (err_before - err_after) gives the game's real px-per-count directly:
    #     true_pxpc = (err_before - err_after) / counts
    # The unknown (how far the target itself moved meanwhile) is a noise term,
    # damped by a running least-squares fit: with counts large the measured
    # slope dominates, with counts small the prior keeps the estimate sane.
    def _update_calibration(self, err):
        """Learn the game's real px-per-count from the last push."""

        def _r(v):
            v = float(v or 0.0)
            return v if math.isfinite(v) else 0.0

        consumed = self._consume_pending()
        if consumed is None:
            return
        sent_cx, sent_cy, pend_err_x, pend_err_y, _pxpc = consumed
        pend_line = (self._pending_line_x, self._pending_line_y)

        cur_offset = self._cur_offset or (0.0, 0.0)
        err_x = _r(cur_offset[0])
        err_y = _r(cur_offset[1])

        kx, ky = self._k_now()
        flipped = False
        learned = 0

        for axis in (0, 1):
            sent = sent_cx if axis == 0 else sent_cy
            before = pend_err_x if axis == 0 else pend_err_y
            after = err_x if axis == 0 else err_y
            if abs(sent) < 3.0 or before is None or abs(before) <= 1e-6:
                continue                      # too small to measure
            if (self._inverted[axis]):
                moved = after - before        # this axis runs the other way
            else:
                moved = before - after        # px actually crossed on this axis
            seed = self.px_per_count_x if axis == 0 else self.px_per_count_y
            along = moved if sent > 0.0 else -moved   # > 0 = the intended way

            # --- inverted axis? we pushed towards, it went further away ---
            # A sign contradiction is unambiguous: an inverted axis moves the
            # crosshair away, while detection noise of a few pixels cannot
            # cancel a step of dozens. One contradiction may still be stale
            # (measured before a previous flip could act), so:
            #   * two in a row while NOT inverted  -> the axis is inverted
            #   * two in a row while inverted      -> the axis is NOT inverted
            #     (the flag no longer matches reality: the game was changed)
            if along < -4.0:
                self._flip_hits[axis] += 1
                # Only a NOT-inverted axis can be proven wrong: while inverted,
                # one success means "not contradicted", never "normal again".
                if self._flip_hits[axis] >= 2 and not self._inverted[axis]:
                    self._inverted[axis] = True
                    self._flip_hits[axis] = 0
                    # direction is new, not confirmed: the step now in flight
                    # was still pushed the old way (see n_sign_known)
                    self._axis_samples[axis] = 0
                flipped = True
                break

            self._flip_hits[axis] = 0
            if along < 1.0:
                continue                      # already on target

            # a real, non-degenerate motion measurement: the sign on this axis
            # is confirmed, so a later step may cover the whole offset
            self._axis_samples[axis] += 1

            # k = true px/count, relative to the config seed
            obs = abs(along / sent) / max(1e-9, seed)
            old = kx if axis == 0 else ky
            obs = min(old * 6.0, max(old * 0.05, obs))

            # A long step is a good measurement (mouse noise is a few counts,
            # the displacement dozens), so it can be trusted almost completely -
            # that is what makes the very first jump land. Short steps are
            # dominated by rounding, so they only refine the estimate.
            trust = 1.0 if abs(sent) >= 8.0 else 0.5
            new = old + trust * (obs - old)

            # --- per-target copy (handles far/near) ---
            # The global factor takes the slow share of the observation, the
            # per-target one the fast share, so a far target is corrected on
            # the spot even while the global factor is still cold.
            line = pend_line[axis]
            target = new if line is None else line + trust * (new - line)
            if axis == 0:
                self._pending_line_x = target
                kx = old + 0.22 * (obs - old)
            else:
                self._pending_line_y = target
                ky = old + 0.22 * (obs - old)
            learned += 1

            if getattr(self, 'debug_calib', False):
                logger.debug(f"[calib] ax{axis} sent={sent:.1f} before={before:.1f} "
                             f"after={after:.1f} obs={obs:.3f} k={old:.3f}->{new:.3f}")

        if flipped:
            # the direction was wrong for every target so far: restart the jump
            self._pending_counts = 0.0
            self._pending_vec = (0.0, 0.0)
            self._pending_err = None
            self._pending_err_x = None
            self._pending_err_y = None
            self._pending_sent = (0.0, 0.0)
            self._pending_line_x = None
            self._pending_line_y = None
            self._aim_samples = 0
            return
        if not learned:
            self._aim_samples += 1
            return

        kx = max(0.05, min(40.0, kx))
        ky = max(0.05, min(40.0, ky))
        self._k_x, self._k_y = kx, ky
        self._aim_scale_seed = 0.5 * (kx + ky)
        # Uncertainty := how far the two independent axes still disagree, plus
        # a floor that drops as observations accumulate. One noisy sample keeps
        # the uncertainty high (small, safe steps); agreeing observations drive
        # it to zero, and then a step covers the whole remaining offset.
        spread = abs(kx - ky) / max(1e-9, 0.5 * (kx + ky))
        floor = 0.30 if self._aim_samples < 2 else 0.02
        self._aim_scale_var = max(spread, floor) ** 2
        self._apply_calibration()
        self._aim_samples += 1

    def _apply_calibration(self):
        """Publish the learned depth factor for the next step."""
        self._aim_depth_scale = getattr(self, '_world_depth', 1.0)

    @property
    def _aim_scale_seed(self):
        """Learned px-per-count factor (1.0 = the config seed was exact)."""
        return getattr(self, '_k_seed', 1.0)

    @_aim_scale_seed.setter
    def _aim_scale_seed(self, value):
        self._k_seed = float(value)

    def _k_now(self):
        """Current global factor per axis (relative to the config seed)."""
        kx = float(getattr(self, '_k_x', self._k_seed) or 1.0)
        ky = float(getattr(self, '_k_y', self._k_seed) or 1.0)
        return kx, ky

    def _k_target(self, axis):
        """Factor to use right now: per-target estimate, else the global one."""
        glob = self._k_now()[axis]
        line = self._pending_line_x if axis == 0 else self._pending_line_y
        if line is None:
            return glob
        return line

    def n_sign_known(self, axis):
        """
        True once this axis' DIRECTION has been established.

        Until then the controller is probing: it does not yet know whether a
        positive count moves the crosshair towards or away from the target, so
        a step landing on the far side is discovery, not overshoot.

        A freshly flipped axis is NOT known yet, even though `_inverted` is
        already set: the step measured in the same frame that revealed the
        flip was still pushed the old way, so one more frame of growth there
        is the tail of the mistake, not a new one. The counter is reset on
        flip and climbs again as soon as the corrected direction is observed.
        """
        axis = 1 if axis else 0
        return self._axis_samples[axis] >= 2

    def _consume_pending(self):
        """
        Return (counts_x, counts_y, err_x, err_y, pxpc) for the push since the
        previous observation, then clear it.

        Several aim ticks fit into one model frame (4 ms vs 29 ms), so the
        pushed steps are ACCUMULATED; a per-axis sum is needed because the
        crosshair travels on both axes independently between two detections.
        """
        if self._pending_sent == (0.0, 0.0) and self._pending_vec == (0.0, 0.0):
            return None
        pending = (float(self._pending_sent[0]), float(self._pending_sent[1]),
                   self._pending_err_x, self._pending_err_y, self._pending_pxpc)
        self._pending_counts = 0.0
        self._pending_vec = (0.0, 0.0)
        self._pending_sent = (0.0, 0.0)
        self._pending_err = None
        self._pending_err_x = None
        self._pending_err_y = None
        self._pending_new_goal = False
        return pending

    def _step_scale(self):
        """
        Fraction of the remaining offset to cover this step.

        The step covers the WHOLE remaining offset as soon as the px-per-count
        seed can be trusted, which is what makes the crosshair land on the
        point in one go instead of creeping towards it frame by frame. While
        the seed is cold the step is softened and the residual clamp below
        keeps the crosshair on this side of the head.
        """
        unc = self._calib_uncertainty()
        gain = 1.0 / (1.0 + unc)
        # The deadzone is handled separately (it IS the lock), so the aim may
        # use the full remaining offset and stop just short of the point.
        # The first steps pay a "research" cost: they are the only way to learn
        # the game's real px-per-count, so they cover a little more than the
        # remaining offset; every step after that lands exactly.
        cover = 1.6 if self._aim_samples < 2 else 1.0
        return max(0.1, min(1.0, gain * max(0.2, min(1.0, self.lock_precise_gain)) * cover))

    def calc_movement(self, target_x, target_y, frame_dt=None, target_h=None):
        """
        Relative aim step towards the detected point. Returns mouse counts.
        (0, 0) => nothing to do: on target (hard lock) or out of zone.
        """
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        dist = math.hypot(offset_x, offset_y)
        self.error_px = dist
        self._cur_offset = (offset_x, offset_y)

        # --- 1. learn from the previous step ---
        # ONLY when a NEW detection arrived. The aim loop runs at ~250 Hz but
        # the model updates every ~29 ms, so between two detections the goal is
        # the same object and the measured distance is unchanged. Learning on
        # those ticks would read "we moved nothing" and push the scale to zero.
        goal = self._goal
        goal_id = goal[6] if goal else 0
        if goal_id and goal_id != self._seen_goal_id:
            self._seen_goal_id = goal_id
            self._update_calibration(dist)

        # no live detection at all -> never move (release on loss)
        if self.release_after <= 0.0 and self._goal is None:
            self.hard_lock = False
            return 0.0, 0.0

        # --- 1b. one push per detection: the result of the counts already
        #         sent is not visible until the next model frame. Repeating
        #         the correction in between would multiply the step by the
        #         number of ticks that fit into a frame (~7), which is the
        #         classic "it jumps past the head" behaviour. ---
        if self._pending_vec != (0.0, 0.0):
            self._in_flight = True
            return 0.0, 0.0
        self._in_flight = False

        # --- target outside the aim zone: leave the player alone ---
        if dist > self.lock_max_zone_px:
            self.hard_lock = False
            self._pending_counts = 0.0
            self._pending_vec = (0.0, 0.0)
            self._pending_err = None
            self._pending_err_x = None
            self._pending_err_y = None
            self._pending_sent = (0.0, 0.0)
            return 0.0, 0.0

        # --- 2. HARD STOP inside the deadzone: send NOTHING ---
        if dist <= self.deadzone_px:
            self.hard_lock = True
            self._pending_counts = 0.0
            self._pending_vec = (0.0, 0.0)
            self._pending_err = None
            self._pending_err_x = None
            self._pending_err_y = None
            self._pending_sent = (0.0, 0.0)
            return 0.0, 0.0

        self.hard_lock = False

        # --- 3. depth (dur / nazdeek) + live calibration ---
        # The box height handed in here is the ONLY distance signal we have:
        # a head twice as far away covers half the pixels, so it needs half
        # the rotation for the same offset. Falling back to 0 (old bug) made
        # every target look "reference distance" and broke far targets.
        if target_h is None:
            goal = self._goal
            target_h = goal[3] if goal else 0.0
        self._cur_target_h = float(target_h or 0.0)
        self._update_depth_reference(self._cur_target_h)
        ratio = self._depth_ratio(target_h)
        # Per axis: global factor x depth x per-target factor. A far target
        # learns its own factor, so it is corrected on the spot instead of
        # waiting for the global estimate to catch up.
        kx = self._k_target(0) * ratio * self._aim_depth_scale
        ky = self._k_target(1) * ratio * self._aim_depth_scale
        px_x = max(1e-4, self.px_per_count_x * kx)
        px_y = max(1e-4, self.px_per_count_y * ky)
        self._target_px_per_count = 0.5 * (px_x + px_y)

        # --- 4. step size: whole offset once calibration is warm ---
        gain = self._step_scale()
        if self.human_moving:
            gain *= 0.6
        self._step_gain = gain

        counts_x = (offset_x * gain) / px_x
        counts_y = (offset_y * gain) / px_y
        # inverted axis (Windows "invert Y", flipped game setting): learned
        # from the observations above, applied from the next step on
        inv = self._inverted
        if inv[0]:
            counts_x = -counts_x
        if inv[1]:
            counts_y = -counts_y

        # --- 6. per-axis: a step sized with a wrong seed must not cross the
        #        head. `upper` is the worst-case px-per-count the estimate can
        #        still have, so counts*upper is the furthest the crosshair could
        #        physically travel; clamping to the remaining offset guarantees
        #        we stop ON the target, never behind it. An additional margin
        #        covers the detection noise of the next box.
        unc = self._calib_uncertainty()
        upper_x = px_x * (1.0 + unc)
        upper_y = px_y * (1.0 + unc)
        margin = max(2.0, min(0.5 * self.deadzone_px, min(self.lock_settle_px, 6.0)))
        room_x = max(0.0, abs(offset_x) - margin)
        room_y = max(0.0, abs(offset_y) - margin)
        safe_x = room_x / max(1e-6, upper_x)
        safe_y = room_y / max(1e-6, upper_y)
        if abs(counts_x) > safe_x:
            counts_x = math.copysign(safe_x, counts_x)
        if abs(counts_y) > safe_y:
            counts_y = math.copysign(safe_y, counts_y)

        # --- 7. magnitude clamp (anti-spike) ---
        # Counts are not pixels: a display coordinate no longer limits them,
        # so the guard has to scale with the real px-per-count of this target,
        # otherwise a far offset is silently truncated and the crosshair never
        # arrives. `lock_max_zone_px` is the actual "too far, do not aim" limit.
        mag = math.hypot(counts_x, counts_y)
        limit = max(2.0 * float(self.lock_max_move_px),
                    (dist * 1.5) / max(1e-6, min(px_x, px_y)))
        if mag > limit and mag > 1e-6:
            counts_x *= limit / mag
            counts_y *= limit / mag

        if self._kal is not None:
            self._kal.update(target_x, target_y)

        # truncate, never round: one count too many is what overshoots
        move_x = int(counts_x)
        move_y = int(counts_y)
        if move_x == 0 and move_y == 0:
            if dist <= max(self.deadzone_px, self.lock_settle_px):
                self.hard_lock = True
            return 0.0, 0.0

        # remember the step so the NEXT observation can learn from it.
        # Several aim ticks fit into one model frame, so the counts are
        # accumulated per axis; the calibration compares them with the
        # per-axis change of the remaining offset.
        if self._pending_err is None:
            # first step of this observation window: remember where we were
            self._pending_err_x = offset_x
            self._pending_err_y = offset_y
        vx, vy = self._pending_vec
        self._pending_vec = (vx + move_x, vy + move_y)
        psx, psy = self._pending_sent
        self._pending_sent = (psx + move_x, psy + move_y)
        self._pending_counts = float(math.hypot(*self._pending_vec))
        self._pending_err = dist
        self._pending_h = self._cur_target_h
        self._pending_pxpc = (px_x, px_y)

        if len(self.aim_log) < 512:
            self.aim_log.append((time.perf_counter(), move_x, move_y, dist))

        return float(move_x), float(move_y)


    def move_mouse(self, move_x, move_y, shooting_state):
        move_x = int(move_x)
        move_y = int(move_y)
        if move_x == 0 and move_y == 0:
            return

        if shooting_state or cfg.mouse_auto_aim:
            with self._hw_lock:
                if cfg.mouse_ghub and hasattr(self, 'ghub') and self.ghub and self.ghub.gmok:
                    self.ghub.mouse_xy(move_x, move_y)
                elif cfg.arduino_move:
                    from logic.arduino import arduino
                    arduino.move(move_x, move_y)
                elif cfg.mouse_rzr:
                    self.rzr.mouse_move(move_x, move_y, True)
                else:
                    input_backend.move_mouse(move_x, move_y)


    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            if input_backend.is_pressed(key_name.strip()):
                return True
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        reduced_w = target_w * reduction_factor / 2
        reduced_h = target_h * reduction_factor / 2
        x1 = target_x - reduced_w
        x2 = target_x + reduced_w
        y1 = target_y - reduced_h
        y2 = target_y + reduced_h
        bScope = self.center_x > x1 and self.center_x < x2 and self.center_y > y1 and self.center_y < y2

        if cfg.show_window and cfg.show_bScope_box:
            visuals.draw_bScope(x1, x2, y1, y2, bScope)

        return bScope

    def update_settings(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        self.arch = self.get_arch()

        self.windows_sens = getattr(cfg, 'windows_sensitivity', 6)
        self.m_yaw = getattr(cfg, 'm_yaw', 0.022)
        self.m_pitch = getattr(cfg, 'm_pitch', 0.022)
        self.deadzone_px = getattr(cfg, 'mouse_deadzone', 4.0)
        self.max_move_per_frame = getattr(cfg, 'max_move_per_frame', 500)

        self.kalman_enabled = KALMAN_AVAILABLE and not cfg.disable_prediction
        self.kalman_lookahead = float(getattr(cfg, 'prediction_interval', 0.045))
        self.snap_jump_enabled = getattr(cfg, 'head_snap_mode', True)
        self.snap_factor = float(getattr(cfg, 'head_snap_factor', 1.0))
        self.pid_enabled = getattr(cfg, 'pid_enabled', False)

        # Magnetic Lock System — re-read config values on settings update
        self.magnetic_strength = float(getattr(cfg, 'magnetic_strength', 4.0))
        self.magnetic_smoothing = float(getattr(cfg, 'magnetic_smoothing', 0.20))
        self.decay_radius = 100.0 / max(1.0, self.magnetic_strength)

        self._init_lock_controller()
        self.reset_filter()
        self._recalc_conversion_factor()
        self.setup_hardware()
        self.start()

    def visualize_target(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.overlay_show_target_line):
            visuals.draw_target_line(target_x, target_y, target_cls)

    def visualize_prediction(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.overlay_show_target_prediction_line):
            visuals.draw_predicted_position(target_x, target_y, target_cls)

    def visualize_history(self, target_x, target_y):
        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.overlay_show_history_points):
            visuals.draw_history_point_add_point(target_x, target_y)

mouse = MouseThread()
