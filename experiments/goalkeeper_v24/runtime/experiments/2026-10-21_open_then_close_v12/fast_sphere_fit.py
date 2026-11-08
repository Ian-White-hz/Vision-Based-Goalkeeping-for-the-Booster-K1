"""Small RGB silhouette fit; no ball state is accepted by this function."""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def fit_center(mask,robot_mask,camera,rotation,initial):
    h,w=mask.shape
    yy,xx=np.nonzero(mask)
    if len(xx)<4: return np.asarray(initial).copy(),1.
    # Keep a margin of background around every observed ball pixel. Faraway
    # background adds no information to this near-ball fit.
    x0,x1=max(0,int(xx.min())-16),min(w,int(xx.max())+17)
    y0,y1=max(0,int(yy.min())-16),min(h,int(yy.max())+17)
    tan=math.tan(math.radians(100)/2)
    u,v=np.meshgrid((np.arange(x0,x1)+.5-w/2)*2*tan/h,-(np.arange(y0,y1)+.5-h/2)*2*tan/h)
    rays=np.stack((u,v,-np.ones_like(u)),2)@rotation.T
    rays/=np.linalg.norm(rays,axis=2,keepdims=True)
    labels=mask[y0:y1,x0:x1]
    keep=~robot_mask[y0:y1,x0:x1]|labels
    directions=rays[keep]; labels=labels[keep].astype(float)
    def residual(xy):
        delta=np.r_[xy,.11]-camera
        along=directions@delta
        perpendicular=np.sqrt(np.maximum(delta@delta-along**2,1e-12))
        return expit((.11-perpendicular)/.0015)*(along>0)-labels
    initial=np.asarray(initial)
    fit=least_squares(residual,initial,bounds=(initial-.08,initial+.08),max_nfev=35,diff_step=1e-4)
    return fit.x,float(np.mean(fit.fun**2))
