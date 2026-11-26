import threading
import time
from typing import List
import cv2
import os

from logic.config_watcher import cfg
from logic.capture import capture
from logic.mouse import mouse
from logic.visual import visuals
from logic.platform import input_backend
from logic.model_classes import (
    HEAD_CLASS_ID,
    HIDEOUT_TARGET_CLASS_IDS,
    PLAYER_CLASS_ID,
)

class HotkeysWatcher(threading.Thread):
    def __init__(self):
        super(HotkeysWatcher, self).__init__()
        self.daemon = True
        self.name = 'HotkeysWatcher'
        
        self.app_pause = 0
        self.auto_mode_enabled = True
        self._f1_prev_state = False
        self._f5_prev_state = False
        self._f6_prev_state = False
        self.clss = self.active_classes()
        self._next_config_poll_at = 0.0

        self.start()
        
    def run(self):
        cfg_reload_prev_state = False
        while True:
            cfg_reload_prev_state = self.process_hotkeys(cfg_reload_prev_state)
            self.reload_config_if_changed()
                
            # terminate
            if input_backend.is_pressed(cfg.hotkey_exit):
                capture.Quit()
                if cfg.show_window:
                    visuals.stop()
                os._exit(0)
            time.sleep(0.01)
            
    def _play_beep(self, freq, duration):
        try:
            import winsound
            winsound.Beep(freq, duration)
        except Exception:
            pass

    def process_hotkeys(self, cfg_reload_prev_state):
        f1_state = input_backend.is_pressed('F1')
        if f1_state and not self._f1_prev_state:
            self.auto_mode_enabled = not self.auto_mode_enabled
            if self.auto_mode_enabled:
                print('\n[AIMBOT] Auto Aim & Fire: ENABLED (ON) [F1]') 
                threading.Thread(target=self._play_beep, args=(1200, 150), daemon=True).start()
            else:
                print('\n[AIMBOT] Auto Aim & Fire: PAUSED (OFF) [F1] - (Detection & Visuals Active)') 
                threading.Thread(target=self._play_beep, args=(500, 200), daemon=True).start()
                try:
                    from logic.frame_parser import frameParser
                    frameParser.reset_lock()
                except Exception:
                    pass
        self._f1_prev_state = f1_state

        # F5: Switch Team (CT <-> T) for Halftime Switch
        f5_state = input_backend.is_pressed('F5')
        if f5_state and not self._f5_prev_state:
            cfg.my_team = "T" if cfg.my_team.upper() == "CT" else "CT"
            cfg.config.set("Team Filter", "my_team", cfg.my_team)
            cfg._save()
            target_team = "TERRORIST (T)" if cfg.my_team.upper() == "CT" else "COUNTER-TERRORIST (CT)"
            print(f"\n[AIMBOT] TEAM SWITCHED! AAPKI TEAM: [{cfg.my_team.upper()}] | DUSHMAN: [{target_team}] [F5]")
            threading.Thread(target=self._play_beep, args=(1100 if cfg.my_team.upper() == "CT" else 800, 150), daemon=True).start()
        self._f5_prev_state = f5_state

        # F6: Toggle Team Filter (ON = Match / OFF = Deathmatch)
        f6_state = input_backend.is_pressed('F6')
        if f6_state and not self._f6_prev_state:
            cfg.team_filter_enabled = not cfg.team_filter_enabled
            cfg.config.set("Team Filter", "team_filter_enabled", "true" if cfg.team_filter_enabled else "false")
            cfg._save()
            status_str = "ENABLED (Protect Teammates)" if cfg.team_filter_enabled else "DISABLED (Attack Everyone / FFA)"
            print(f"\n[AIMBOT] TEAM FILTER: {status_str} [F6]")
            threading.Thread(target=self._play_beep, args=(1350 if cfg.team_filter_enabled else 600, 150), daemon=True).start()
        self._f6_prev_state = f6_state

        self.app_pause = -1 if input_backend.is_pressed(cfg.hotkey_pause) else 0
        app_reload_cfg = input_backend.is_pressed(cfg.hotkey_reload_config)
        
        if app_reload_cfg and not cfg_reload_prev_state:
            cfg.Read(verbose=True)
            self.apply_config_changes()
                    
        return app_reload_cfg

    def reload_config_if_changed(self):
        now = time.monotonic()
        if now < self._next_config_poll_at:
            return
        self._next_config_poll_at = now + 0.25
        if cfg.reload_if_changed(verbose=True):
            self.apply_config_changes()

    def apply_config_changes(self):
        capture.restart()
        mouse.update_settings()
        try:
            from logic.frame_parser import frameParser
            frameParser.update_settings()
        except ImportError:
            pass
        self.clss = self.active_classes()
        if cfg.show_window == False:
            cv2.destroyAllWindows()

    def active_classes(self) -> List[int]:
        clss = [PLAYER_CLASS_ID]
        
        if cfg.hideout_targets:
            clss.extend(HIDEOUT_TARGET_CLASS_IDS)

        if not cfg.disable_headshot:
            clss.append(HEAD_CLASS_ID)
            
        self.clss = sorted(set(clss))
        return self.clss
    
hotkeys_watcher = HotkeysWatcher()
