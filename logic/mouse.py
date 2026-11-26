import time
import math
import os
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
    Aim control.

    WHY THIS IS RELATIVE-ONLY
    -------------------------
    Research / verified behaviour: an FPS reads mouse input through Raw Input
    (WM_INPUT / DirectInput), NOT through the OS cursor. So SetCursorPos /
    MOUSEEVENTF_ABSOLUTE move the *desktop cursor* only and the game view does
    not follow at all (that is why the absolute build looked like "mouse hi
    nahi hilta"). Games only react to *relative* deltas (MOUSEEVENTF_MOVE),
    which is exactly what this class sends.

    WHY THE OLD BUILD OVERSHOT
    --------------------------
    Old code:  move = offset_px * counts_per_px   (one open-loop guess per frame)
      * counts_per_px came from sens x yaw x FOV and is only an ESTIMATE,
      * so after one frame a residual offset always remained,
      * next frame that residual produced ANOTHER delta -> the crosshair
        glided past the detected point (overshoot / no lock).
    There is no feedback loop, so nothing ever says "you arrived, stop".

    HOW THIS BUILD GUARANTEES "land and stop"
    -----------------------------------------
    1. CALIBRATION: the aim unit is corrected live by what the game actually
       does. Every step we remember (delta sent) and compare it with the
       measured change in the target offset. ratio = measured / sent, so the
       conversion converges to the real in-game sensitivity by itself — no
       config numbers need to be right.
    2. CLOSED LOOP: each frame we re-measure offset from the CURRENT detected
       position to the crosshair, so we always steer from truth, never from a
       guess.
    3. NO-OVERSHOOT GAIN: the step never exceeds the remaining offset
       (scale <= 1.0 always). Higher gain near the target = faster arrival,
       lower gain = smooth. Overshoot is mathematically impossible.
    4. HARD STOP: once inside `mouse_deadzone` px we send NOTHING for
       `lock_dwell_frames` frames. The crosshair physically stays put.
    5. If the loop detects the crosshair went PAST the target (overshoot),
       calibration is corrected and the next step aims back — self-healing,
       regardless of DPI / sens / m_yaw.
    """

    def __init__(self):
        self.initialize_parameters()
        self.setup_hardware()

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
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.prev_distance = None
        self.speed_correction_factor = 0.1
        self.bScope = False
        self.arch = self.get_arch()
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100

        self.windows_sens = getattr(cfg, 'windows_sensitivity', 6)
        self.m_yaw = getattr(cfg, 'm_yaw', 0.022)
        self.m_pitch = getattr(cfg, 'm_pitch', 0.022)
        self.deadzone_px = getattr(cfg, 'mouse_deadzone', 2.0)
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

        try:
            from simple_pid import PID
            self.pid_x = PID(
                Kp=self.pid_kp, Ki=self.pid_ki, Kd=self.pid_kd,
                setpoint=0.0, sample_time=None,
                differential_on_measurement=True,
                output_limits=(-float(self.max_move_per_frame), float(self.max_move_per_frame))
            )
            self.pid_y = PID(
                Kp=self.pid_kp, Ki=self.pid_ki, Kd=self.pid_kd,
                setpoint=0.0, sample_time=None,
                differential_on_measurement=True,
                output_limits=(-float(self.max_move_per_frame), float(self.max_move_per_frame))
            )
        except Exception:
            self.pid_x = None
            self.pid_y = None

        self.last_pid_time = time.perf_counter()

        # === MAGNETIC LOCK (relative, closed loop) ===
        self.magnetic_strength = float(getattr(cfg, 'magnetic_strength', 4.0))
        self.magnetic_smoothing = float(getattr(cfg, 'magnetic_smoothing', 0.20))
        self.decay_radius = 100.0 / max(1.0, self.magnetic_strength)

        self._recalc_conversion_factor()
        self._init_lock_controller()

    # ------------------------------------------------------------------ utils
    def _init_lock_controller(self):
        """Live-calibrating, non-overshooting relative aim controller state."""
        self.lock_max_zone_px = float(getattr(cfg, 'lock_max_zone_px', 480.0))
        self.lock_max_move_px = float(getattr(cfg, 'lock_max_move_px', 140.0))
        self.lock_settle_px = float(getattr(cfg, 'lock_settle_px', 3.0))
        self.lock_dwell_frames = max(1, int(getattr(cfg, 'lock_dwell_frames', 6)))

        # Snap (flick) tuning — replaces the old blind 1.0 multiplier
        self.lock_snap_px = float(getattr(cfg, 'lock_snap_px', 90.0))
        self.lock_snap_gain = float(getattr(cfg, 'lock_snap_gain', 1.0))
        self.lock_precise_gain = float(getattr(cfg, 'lock_precise_gain', 0.75))

        # Worst-case sensitivity error the first steps must survive. The
        # config estimate can be 4x off in either direction (different game,
        # different scope, changed in-game sens), so the very first step is
        # clamped to offset/4. It is NOT a tuning knob you have to get right:
        # any value >= 1.5 stays safe, it only affects how many frames the
        # first approach takes.
        self.lock_sens_guard = float(getattr(cfg, 'lock_sens_guard', 5.0))

        # Set by process_data() before calc_movement()
        self.human_moving = False
        self._reset_loop()

    def _reset_loop(self):
        """Forget calibration + pending step (new target / settings reload)."""
        self.hard_lock = False
        self.lock_dwell_left = 0
        self._samples = 0
        self._pending = None
        self._prev_offset = None
        self._last_axis_offset = None
        self._cur_offset = None
        # live calibration seeds come from the config estimate; _update_calibration
        # then converges them to the real in-game behaviour
        self._px_count_x = float(getattr(self, 'px_per_count_x', 1.0))
        self._px_count_y = float(getattr(self, 'px_per_count_y', 1.0))
        self._invert_x = False
        self._invert_y = False
        self._cur_ratio = None
        self._last_ratio = None
        self._pending_expected = None
        self._human_hits = 0

    def reset_filter(self):
        self.filter_x = None
        self.filter_y = None
        self.kalman_tracker = None
        self.first_target_seen = False
        self.locked_on_target = False
        self.consecutive_lock_frames = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.prev_distance = None
        self._reset_loop()
        if hasattr(self, 'pid_x') and self.pid_x is not None:
            self.pid_x.reset()
        if hasattr(self, 'pid_y') and self.pid_y is not None:
            self.pid_y.reset()
        self.last_pid_time = time.perf_counter()

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
            fov_x = 90.0
            fov_y = 73.74
        else:
            fov_x = max(1.0, float(self.fov_x))
            fov_y = max(1.0, float(self.fov_y))

        deg_per_px_x = fov_x / disp_w
        deg_per_px_y = fov_y / disp_h

        base_counts_x = deg_per_px_x / deg_per_count_x
        base_counts_y = deg_per_px_y / deg_per_count_y
        calib_scale = float(getattr(cfg, 'mouse_calib_scale', 1.0))

        self.counts_per_px_x = base_counts_x * calib_scale
        self.counts_per_px_y = base_counts_y * calib_scale
        self.px_per_count_x = 1.0 / max(1e-5, self.counts_per_px_x)
        self.px_per_count_y = 1.0 / max(1e-5, self.counts_per_px_y)
        self.display_w = disp_w
        self.display_h = disp_h

        logger.info(f"[Mouse] counts/px: x={self.counts_per_px_x:.4f} y={self.counts_per_px_y:.4f} | FOV: {fov_x}x{fov_y} | Disp: {disp_w}x{disp_h} | calib={calib_scale}")

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

    def _get_screen_rect(self):
        """
        Screen region the detection window represents, in physical pixels:
        (left, top, width, height, scale_x, scale_y).
        Detection px -> screen px scaling is needed because the capture window
        (320x320) is usually smaller than the real screen (e.g. 1920x1080).
        """
        scr_x, scr_y, scr_w, scr_h = 0, 0, int(self.display_w), int(self.display_h)
        try:
            from logic.capture import capture
            left, top, right, bottom = capture.calculate_screen_offset()
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            if right > left and bottom > top:
                scr_x, scr_y = left, top
                scr_w, scr_h = right - left, bottom - top
                self._capture_rect = (scr_x, scr_y, scr_w, scr_h)
        except Exception as exc:
            cached = getattr(self, '_capture_rect', None)
            if cached:
                scr_x, scr_y, scr_w, scr_h = cached
            else:
                logger.debug(f"[Mouse] screen rect fallback: {exc}")

        det_w = self.screen_width or 320
        det_h = self.screen_height or 320
        scale_x = scr_w / max(1.0, det_w)
        scale_y = scr_h / max(1.0, det_h)
        return scr_x, scr_y, scr_w, scr_h, scale_x, scale_y

    # ------------------------------------------------------------- detection
    def process_data(self, data):
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
        self.human_moving = self._detect_human_movement(target_x, target_y)
        is_head = is_head_class(target_cls) if target_cls is not None else False

        self.visualize_target(target_x, target_y, target_cls)

        self.bScope = self.check_target_in_scope(
            target_x, target_y, target_w, target_h, self.bScope_multiplier
        ) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        # Aim point: Target() already lifted body boxes by body_y_offset,
        # so the value handed here is the point the crosshair should sit on.
        move_x, move_y = self.calc_movement(target_x, target_y)

        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        dist = math.hypot(offset_x, offset_y)

        self.visualize_history(target_x, target_y)
        shooting.submit(self.bScope, shooting_state)
        self.move_mouse(move_x, move_y, shooting_state)

        if self.hard_lock:
            self.consecutive_lock_frames += 1
            self.locked_on_target = True
        else:
            self.consecutive_lock_frames = 0
            self.locked_on_target = False

        logger.debug(
            f"[Mouse] target=({target_x:.0f},{target_y:.0f}) offset=({offset_x:+.0f},{offset_y:+.0f}) "
            f"dist={dist:.0f}px move=({move_x:+.1f},{move_y:+.1f}) "
            f"calib=({self._px_count_x:.3f},{self._px_count_y:.3f})px/cnt locked={self.locked_on_target}"
        )

    def _detect_human_movement(self, target_x, target_y):
        """
        Did the PLAYER move the mouse this frame, or was it all our own step?

        Comparing raw target positions is wrong: during a fast approach the
        offset legitimately shrinks by 20-90 px per frame because WE moved it.
        That made this fire mid-approach and throttle the gain to 0.45x for no
        reason (visible as the aim slowing down just before it locks).

        So subtract what our last step was supposed to achieve. Only the part
        of the change our step cannot explain is the player's own input.
        """
        if self._last_axis_offset is None:
            return False
        cur_x = target_x - self.center_x
        cur_y = target_y - self.center_y
        # What our last step should have removed, using the LIVE px/count.
        exp_x = exp_y = 0.0
        if self._pending_expected is not None:
            cnt_x, cnt_y = self._pending_expected
            exp_x = cnt_x * self._px_count_x
            exp_y = cnt_y * self._px_count_y
        residual_x = abs((cur_x - self._last_axis_offset[0]) - exp_x)
        residual_y = abs((cur_y - self._last_axis_offset[1]) - exp_y)
        limit = float(getattr(cfg, 'lock_human_px_per_frame', 45.0))

        # Only a LARGE impossible disturbance counts as the player fighting us.
        # Small residuals are always our own calibration error: treating those
        # as "human input" throttled the gain and made the aim crawl towards
        # the target instead of snapping onto it.
        # Scale the threshold by how uncertain the calibration still is.
        unc = self._calib_uncertainty()
        allowed_x = limit + abs(exp_x) * unc * 0.5
        allowed_y = limit + abs(exp_y) * unc * 0.5
        return (residual_x > allowed_x * 2.0) or (residual_y > allowed_y * 2.0)

    # ---------------------------------------------------------------- control
    def _distance_gain(self, dist_px):
        """
        Fraction of the remaining offset to move this frame, in (0, 1].

        The arrival is deliberately progressive, not "jump the whole offset":
          * far away      -> approaching gain (fast approach, still < 1.0)
          * near the point -> small fraction, so the last pixels are covered
            with small steps and the crosshair settles instead of snapping
            through the target,
          * once the calibration is still fresh (few samples) the gain is
            reduced, because the px/count estimate may still be wrong.
        """
        if dist_px <= self.lock_settle_px:
            return 0.0

        # 1) base curve: 1 - exp(-d/k) keeps it always < 1.0 and gives
        #    big steps when far, tiny steps when close (no overshoot possible)
        k = max(6.0, self.lock_snap_px * 0.6)
        curve = 1.0 - math.exp(-dist_px / k)
        gain = self.lock_precise_gain + (1.0 - self.lock_precise_gain) * curve
        gain *= max(0.05, min(1.0, self.lock_snap_gain))

        # 2) magnetic strength: snappier pull when the user raised it
        if self.magnetic_strength > 0.0:
            mag = 1.0 - math.exp(-dist_px / max(8.0, self.decay_radius))
            gain *= 0.55 + 0.45 * mag

        # 3) trust: until the live calibration has a few samples, don't gamble
        trust = min(1.0, 0.35 + 0.65 * (self._samples / 4.0))
        gain *= trust

        # 4) final approach: never more than this fraction of what is left
        if dist_px <= max(self.lock_settle_px * 3.0, self.deadzone_px * 2.0):
            gain = min(gain, 0.5)

        return max(0.03, min(0.98, gain))

    def _calib_uncertainty(self):
        """
        How wrong may _px_count be, as a fraction of itself?

        The stored estimate is always one frame behind reality (it is updated
        from the step that already happened). Sizing a step with the point
        estimate therefore guarantees occasional overshoot. This returns a
        safety margin that decays with evidence:
          * no samples yet      -> wide (the config number may be 4x off)
          * sample count grows  -> narrower
          * last observations agree -> narrower still
        """
        n = self._samples
        if n <= 0:
            return max(0.0, float(getattr(self, 'lock_sens_guard', 5.0)) - 1.0)
        # An EMA with alpha=0.5 has residual geometric error: after n updates
        # the estimate is still off by roughly (0.5**n) of the initial gap.
        # That is the term that keeps biting in the final frames, because a
        # couple of agreeing samples makes `spread` look reassuringly small
        # while the estimate is quietly 2-3% short.
        convergence = 0.5 ** n
        # disagreement between the two most recent ratios widens the margin
        prev = getattr(self, '_last_ratio', None)
        cur = getattr(self, '_cur_ratio', None)
        if prev and cur and prev > 1e-6 and cur > 1e-6:
            spread = abs(cur - prev) / max(prev, cur)
        else:
            spread = 0.5
        return max(0.05, min(1.5, convergence + 0.5 * spread))

    def _update_calibration(self):
        """
        Learn the REAL px-per-count of the game from the last step.

        We know what we sent (counts) and how much the offset actually changed
        (screen px).  observed = delta_offset / sent_counts  =>  px per count.
        Folding that into the estimate makes the NEXT step land on the point
        instead of short / long. This feedback loop is what the old open-loop
        build never had (it just trusted the config numbers forever).
        """
        pend = self._pending
        self._pending = None
        if pend is None or self._cur_offset is None:
            return

        sent_x, sent_y, prev_ox, prev_oy = pend
        now_ox, now_oy = self._cur_offset

        # offset change produced by OUR step (screen px)
        delta_x = prev_ox - now_ox
        delta_y = prev_oy - now_oy

        # constant bias: if the game does not consume all counts (rounding,
        # internal scaling) the observed px is slightly less than asked. Model
        # it as delta = sent * k + bias and it converges to the true mapping.
        self._learn_bias('x', sent_x, delta_x)
        self._learn_bias('y', sent_y, delta_y)

        alpha = 0.5
        old_x, old_y = self._px_count_x, self._px_count_y
        self._px_count_x = self._learn_axis(
            'x', sent_x, delta_x, self._px_count_x,
            float(getattr(self, 'px_per_count_x', 1.0)), alpha)
        self._px_count_y = self._learn_axis(
            'y', sent_y, delta_y, self._px_count_y,
            float(getattr(self, 'px_per_count_y', 1.0)), alpha)

        # track how much each observation moved the estimate: a jump means we
        # still do not understand this game's mapping, so stay conservative
        shift = 0.0
        for old, new in ((old_x, self._px_count_x), (old_y, self._px_count_y)):
            if abs(old) > 1e-6:
                shift = max(shift, abs(new - old) / abs(old))
        self._last_ratio = getattr(self, '_cur_ratio', None)
        self._cur_ratio = max(1e-6, shift)
        self._samples += 1

    def _learn_bias(self, axis, sent, delta):
        """Track the constant px the game adds/loses per step (rounding bias)."""
        if abs(sent) < 2.0 or abs(delta) <= 0.3:
            return
        k = abs(self._px_count_x if axis == 'x' else self._px_count_y)
        expected = abs(sent) * k
        observed = abs(delta)
        err = observed - expected
        cur = getattr(self, f'_bias_{axis}', 0.0)
        # trust slowly, clamp so a bad sample can not wreck the loop
        cur = 0.7 * cur + 0.3 * err
        limit = max(3.0, expected * 0.5)
        setattr(self, f'_bias_{axis}', max(-limit, min(limit, cur)))

    def _learn_axis(self, axis, sent, delta, current, seed, alpha):
        if abs(sent) < 2.0 or abs(delta) <= 0.5:
            return current

        observed = delta / sent                 # signed px per count
        inverted = getattr(self, f'_invert_{axis}')
        magnitude = abs(current)
        # A positive magnitude means: "+1 count (after flip) moves the view
        # towards positive screen offset". Sign tells how it is mapped.
        sign = -1.0 if inverted else 1.0

        if observed * sign <= 0:
            # View moved opposite to what we asked for.
            if abs(observed) > 0.9 * magnitude:
                # ~right size, wrong direction -> this axis is inverted.
                setattr(self, f'_invert_{axis}', not inverted)
                return abs(observed)
            if abs(observed) < 0.65 * magnitude:
                # Ended up further away on the same side -> overshoot.
                return max(1e-4, magnitude * 0.5)
            return magnitude

        # Same direction: correct the magnitude, keep the mapping.
        target = abs(observed)
        new = (1.0 - alpha) * magnitude + alpha * target
        return max(1e-4, min(3.0, new))

        # Normal correction (signed, respects the current flip state)
        target = abs(observed) * sign
        new = (1.0 - alpha) * current + alpha * target
        return max(1e-4, min(3.0, new))

    def calc_movement(self, target_x, target_y):
        """
        Relative, non-overshooting, self-calibrating aim step.

        Returns mouse counts to send. (0, 0) => do nothing: the crosshair is
        already on the detected point (hard lock), so it cannot drift past it.
        """
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        dist = math.hypot(offset_x, offset_y)

        # remember current truth for next frame's calibration
        self._cur_offset = (offset_x, offset_y)
        self._last_axis_offset = (offset_x, offset_y)

        # --- Target outside the magnetic zone: leave the player alone ---
        if dist > self.lock_max_zone_px:
            self.hard_lock = False
            self.lock_dwell_left = 0
            self._pending = None
            self._prev_offset = (offset_x, offset_y)
            return 0.0, 0.0

        # --- 1. learn from the previous frame's step ---
        self._update_calibration()
        self._prev_offset = (offset_x, offset_y)

        # --- 2. HARD STOP inside the deadzone: send NOTHING ---
        if dist <= self.deadzone_px:
            self.hard_lock = True
            self.lock_dwell_left = self.lock_dwell_frames
            return 0.0, 0.0

        # hold position for a few frames after arriving
        if self.lock_dwell_left > 0:
            self.lock_dwell_left -= 1
            self.hard_lock = True
            return 0.0, 0.0

        self.hard_lock = False

        # --- 3. gain: progressive arrival, always < 1.0 (no overshoot) ---
        gain = self._distance_gain(dist)

        if self.human_moving:
            gain *= 0.45   # player is flicking -> assist, don't hijack

        # --- 4. screen px -> mouse counts (live calibration) ---
        px_x = max(1e-4, self._px_count_x)
        px_y = max(1e-4, self._px_count_y)
        # bias compensation: the game always delivers a bit less px than the
        # counts asked for (rounding / internal scaling). Asking for the extra
        # amount makes the final position converge to zero instead of stalling
        # a few px short.
        want_x = offset_x * gain
        want_y = offset_y * gain
        counts_x = want_x / px_x
        counts_y = want_y / px_y
        if self._samples >= 2:
            bias_x = getattr(self, '_bias_x', 0.0)
            bias_y = getattr(self, '_bias_y', 0.0)
            if 0.0 < abs(bias_x) < max(6.0, abs(want_x) * 0.6):
                counts_x *= 1.0 + bias_x / max(1e-6, abs(want_x))
            if 0.0 < abs(bias_y) < max(6.0, abs(want_y) * 0.6):
                counts_y *= 1.0 + bias_y / max(1e-6, abs(want_y))

        # --- 5a. FIRST-CONTACT CAP ---------------------------------------
        # While the calibration is cold we do NOT know the real px-per-count.
        # The config estimate can be off by 4x, and both clamps below compare
        # against that same wrong estimate, so neither can save us: that is
        # exactly the step that used to fly PAST the detected point.
        #
        # Key idea: bound the step by the WORST-CASE travel it could produce,
        # not by the estimate. If px-per-count could really be `sens_err`
        # times larger, then `n` counts may travel up to `n * px * sens_err`.
        # Requiring that bound to stay below the remaining distance makes an
        # overshoot impossible even though we are guessing.
        if self._samples < 3:
            sens_err = max(1.5, float(getattr(self, 'lock_sens_guard', 5.0)))
            worst_x = abs(counts_x) * px_x * sens_err
            worst_y = abs(counts_y) * px_y * sens_err
            worst = math.hypot(worst_x, worst_y)
            # allow a fraction that grows as evidence arrives, but never
            # enough to cross the target under the worst-case assumption
            relax = (1.0 + self._samples) / 3.0
            budget = dist * min(1.0, relax) * 0.95
            if worst > budget and worst > 1e-6:
                k = budget / worst
                counts_x *= k
                counts_y *= k
                want_x *= k
                want_y *= k

        # --- 5b. magnitude clamp in SCREEN px: never cross the target -----
        # We must reason with an UPPER BOUND on travel, not the point
        # estimate: the estimate always lags the truth by one frame, and a
        # step sized against a too-small estimate is exactly what hops over
        # the head. px_upper = estimate * (1 + uncertainty).
        unc = self._calib_uncertainty()
        upper_x = px_x * (1.0 + unc)
        upper_y = px_y * (1.0 + unc)
        worst_mag = math.hypot(counts_x * upper_x, counts_y * upper_y)
        if worst_mag > dist and worst_mag > 1e-6:
            k = dist / worst_mag
            counts_x *= k
            counts_y *= k

        # --- 6. count magnitude clamp (anti-spike / anti-flick-through) ---
        mag = math.hypot(counts_x, counts_y)
        limit = float(self.lock_max_move_px)
        if mag > limit and mag > 1e-6:
            counts_x *= limit / mag
            counts_y *= limit / mag

        # inverted axis (classic "inverted Y" / flipped coords): flip counts
        if getattr(self, '_invert_x', False):
            counts_x = -counts_x
        if getattr(self, '_invert_y', False):
            counts_y = -counts_y

        # --- 7. minimum step: actually deliver the last pixel, no stalling ---
        # A single count is the smallest thing we can send. If ONE count
        # already moves further than what is left on that axis, sending it
        # would hop over the target — so we must stay silent instead of
        # "trying to be precise". This is what caused the final +/-1 px
        # flicker onto the wrong side of the head.
        if abs(counts_x) < 1.0 and abs(counts_y) < 1.0:
            if dist <= max(4.0, self.lock_settle_px * 2.0):
                if abs(offset_x) >= 0.6 and px_x <= abs(offset_x):
                    counts_x = math.copysign(1.0, offset_x) * (-1.0 if getattr(self, "_invert_x", False) else 1.0)
                if abs(offset_y) >= 0.6 and px_y <= abs(offset_y):
                    counts_y = math.copysign(1.0, offset_y) * (-1.0 if getattr(self, "_invert_y", False) else 1.0)
            else:
                return 0.0, 0.0

        # --- 8. FINAL per-axis sanity check on the rounded step -----------
        # Everything above works with fractional counts; the wire takes ints,
        # and rounding 0.6 up to 1 is exactly how a "safe" step turns into a
        # hop over the head. Round with `floor` on the safe magnitude instead,
        # and drop the axis entirely if not even one count fits.
        safe_x = abs(offset_x) / max(1e-6, px_x * (1.0 + self._calib_uncertainty()))
        safe_y = abs(offset_y) / max(1e-6, px_y * (1.0 + self._calib_uncertainty()))
        if abs(counts_x) > safe_x:
            counts_x = math.copysign(min(abs(counts_x), math.floor(safe_x)), counts_x)
        if abs(counts_y) > safe_y:
            counts_y = math.copysign(min(abs(counts_y), math.floor(safe_y)), counts_y)

        move_x = int(round(counts_x))
        move_y = int(round(counts_y))
        if move_x == 0 and move_y == 0:
            return 0.0, 0.0

        # store for next frame's calibration
        self._pending = (move_x, move_y, offset_x, offset_y)
        # the raw COUNTS we sent: next frame multiplies them by the live
        # px/count to know what this step should have achieved
        self._pending_expected = (move_x, move_y)
        return float(move_x), float(move_y)


    def move_mouse(self, move_x, move_y, shooting_state):
        move_x = int(move_x)
        move_y = int(move_y)
        if move_x == 0 and move_y == 0:
            return

        if shooting_state or cfg.mouse_auto_aim:
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
        self.prev_time = None
        self.prev_distance = None

        self.windows_sens = getattr(cfg, 'windows_sensitivity', 6)
        self.m_yaw = getattr(cfg, 'm_yaw', 0.022)
        self.m_pitch = getattr(cfg, 'm_pitch', 0.022)
        self.deadzone_px = getattr(cfg, 'mouse_deadzone', 2.0)
        self.max_move_per_frame = getattr(cfg, 'max_move_per_frame', 500)

        self.kalman_enabled = KALMAN_AVAILABLE and not cfg.disable_prediction
        self.kalman_lookahead = float(getattr(cfg, 'prediction_interval', 0.045))
        self.snap_jump_enabled = getattr(cfg, 'head_snap_mode', True)
        self.snap_factor = float(getattr(cfg, 'head_snap_factor', 1.0))
        self.pid_enabled = getattr(cfg, 'pid_enabled', False)

        # Magnetic Lock System — re-read config values on settings update
        self.magnetic_strength = float(getattr(cfg, 'magnetic_strength', 4.0))
        self.magnetic_smoothing = float(getattr(cfg, 'magnetic_smoothing', 0.20))
        self.deadzone_px = float(getattr(cfg, 'mouse_deadzone', 2.0))
        self.decay_radius = 100.0 / max(1.0, self.magnetic_strength)

        self._init_lock_controller()
        self.reset_filter()
        self._recalc_conversion_factor()
        self.setup_hardware()

    def visualize_target(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.overlay_show_target_line):
            visuals.draw_target_line(target_x, target_y, target_cls)

    def visualize_prediction(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.overlay_show_target_prediction_line):
            visuals.draw_predicted_position(target_x, target_y, target_cls)

    def visualize_history(self, target_x, target_y):
        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(target_x, target_y)

mouse = MouseThread()
