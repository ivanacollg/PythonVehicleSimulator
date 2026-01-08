#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guidance algorithms.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
import math

# [x_d,v_d,a_d] = refModel3(x_d,v_d,a_d,r,wn_d,zeta_d,v_max,sampleTime) is a 3-order 
# reference  model for generation of a smooth desired position x_d, velocity |v_d| < v_max, 
# and acceleration a_d. Inputs are natural frequency wn_d and relative damping zeta_d.
def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime):
    
    # desired "jerk"
    j_d = wn_d**3 * (r -x_d) - (2*zeta_d+1) * wn_d**2 * v_d - (2*zeta_d+1) * wn_d * a_d

   # Forward Euler integration
    x_d += sampleTime * v_d             # desired position
    v_d += sampleTime * a_d             # desired velocity
    a_d += sampleTime * j_d             # desired acceleration 
    
    # Velocity saturation
    if (v_d > v_max):
        v_d = v_max
    elif (v_d < -v_max): 
        v_d = -v_max    
    
    return x_d, v_d, a_d

def refModel2(v_d, a_d, v_target, wn_d, zeta_d, v_max, sampleTime):
    """
    2nd-order Reference Model for Surge Speed.
    Clamps the OUTPUT speed (u_d), similar to the 3rd-order model.
    """
    
    # 1. Calculate the "jerk" (change in acceleration)
    # Dynamics: mass-spring-damper for speed
    da_d = wn_d**2 * (v_target - v_d) - 2 * zeta_d * wn_d * a_d
    
    # 2. Forward Euler integration
    v_d += sampleTime * a_d  # Update speed
    a_d += sampleTime * da_d # Update acceleration
    
    # 3. Output Saturation (The part you requested)
    if v_d > v_max:
        v_d = v_max
        a_d = 0.0  # <--- CRITICAL: Stop accelerating if we hit the wall!
        
    elif v_d < -v_max:
        v_d = -v_max
        a_d = 0.0  # <--- CRITICAL: Stop accelerating if we hit the wall!
        
    return v_d, a_d