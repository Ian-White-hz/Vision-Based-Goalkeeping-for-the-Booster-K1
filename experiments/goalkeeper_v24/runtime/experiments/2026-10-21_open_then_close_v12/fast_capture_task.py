"""Four-second descent and earlier bounded arm feedback for residual rebound.

The learned gait/pose, scene and real hold gates stay the same. Arm feedback
still requires measured stop, real foot support, low root speeds and RGB fit.
"""
from grip_hold_task import GripIKTouchEnv,make_cfg,measurements,raw_class
from touch_task import TouchIncomingEnv
from bridge import phase


class FastCaptureEnv(GripIKTouchEnv):
    descent_rate=1.5  # Six-second reference traversed in four real seconds.
    servo_phase_min=.85

    def adjust_motor_targets(self,target):
        if self.env.common_step_counter%5==0:
            self.capture_pose_complete=(phase(self.env)>=.98).cpu().numpy()
        return super().adjust_motor_targets(target)

    def update_hand_radius(self,index,forces):
        # Track/contact during the final bend, but establish the fixed RGB/tactile
        # hold anchor only when the posture is almost complete.
        if not self.capture_pose_complete[index] and not self.grip_hold[index]:
            self.grip_contact_dwell[index]=0.
            return TouchIncomingEnv.update_hand_radius(self,index,forces)
        return super().update_hand_radius(index,forces)


env_class=FastCaptureEnv
