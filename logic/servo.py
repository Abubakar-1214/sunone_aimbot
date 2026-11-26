"""
Servo module placeholder - background thread disabled to prevent async loop desync.
Tracking & 1-Euro Filter are now directly managed inside MouseThread for synchronous precision.
"""
class NullServo:
    def __init__(self):
        self.is_alive = lambda: False
        self.target_x = 160.0
        self.target_y = 160.0
    def set_calibration(self, *args, **kwargs): pass
    def update_target(self, *args, **kwargs): pass
    def clear_target(self): pass
    def is_locked(self): return False

servo = NullServo()
