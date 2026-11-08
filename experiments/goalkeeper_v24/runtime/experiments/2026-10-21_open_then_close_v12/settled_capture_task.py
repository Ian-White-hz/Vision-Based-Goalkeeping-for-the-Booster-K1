"""Reduce tactile hold hunting without changing real success requirements.

Recorded failure 26 alternated hand loss while the old force loop repeatedly
opened a secure contact. Use a wider control deadband and bounded small radius
updates after the existing RGB/tactile anchor is established. Acceptance still
requires actual bilateral contact and two continuous real seconds on ground.
"""
import numpy as np
from fast_capture_task import FastCaptureEnv,make_cfg,measurements,raw_class


class SettledCaptureEnv(FastCaptureEnv):
    def update_hand_radius(self,index,forces):
        if not bool(self.grip_hold[index]):
            return super().update_hand_radius(index,forces)
        self.grip_release_dwell[index]=self.grip_release_dwell[index]+.1 if (forces<.05).any() else 0.
        if self.grip_release_dwell[index]>=.299:
            self.grip_hold[index]=False
            self.grip_contact_dwell[index]=self.grip_release_dwell[index]=0.
            return super().update_hand_radius(index,forces)
        # This is a control deadband, not a change to contact/hold acceptance.
        # A tiny commanded radius increment limits lag-driven force oscillation.
        change=np.where(forces<.6,forces-.6,np.where(forces>2.5,forces-2.5,0.))*.0001
        change=np.clip(change,-.00005,.00005)
        # Keep a separate faster pressure release for excessive measured force.
        change=np.where(forces>4.,np.minimum((forces-4.)*.0002,.00025),change)
        self.hand_radius[index]=np.clip(self.hand_radius[index]+change,.129,.137)


class QuickCaptureEnv(SettledCaptureEnv):
    # A separate trial; 3-second reference-only screens already had 0/18 falls.
    # Full standing-start rolling-shot acceptance is still required.
    descent_rate=2.


env_class=SettledCaptureEnv
