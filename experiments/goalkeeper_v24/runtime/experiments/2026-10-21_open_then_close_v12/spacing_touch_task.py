"""Preserve bimanual clearance while applying grounded RGB/tactile contact."""
import numpy as np
from touch_task import TouchIncomingEnv,make_cfg,measurements


class SpacingTouchEnv(TouchIncomingEnv):
    def desired_hand_position(self,ball,delta,radius,side,index=None):
        offset=delta/np.linalg.norm(delta)*radius
        # Hand-end spheres have 25mm radius. Keep 60mm between their centers,
        # then choose the upper contact surface, instead of squeezing the two
        # hands into each other when the ball stops close between the feet.
        offset[1]=(1. if side==0 else -1.)*max(abs(offset[1]),.030)
        height2=radius**2-offset[0]**2-offset[1]**2
        if height2>.0025: offset[2]=np.sqrt(height2)
        return ball+offset
