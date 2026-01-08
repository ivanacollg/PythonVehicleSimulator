#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blueboat.py: 
    Class for the Maritime Robotics blueboat USV, www.maritimerobotics.com. 
    The length of the USV is L = 2.0 m. The constructors are:

    blueboat()                                          
        Step inputs for propeller revolutions n1 and n2
        
    blueboat('headingAutopilot',psi_d,V_current,beta_current,tau_X)  
       Heading autopilot with options:
          psi_d: desired yaw angle (deg)
          V_current: current speed (m/s)
          beta_c: current direction (deg)
          tau_X: surge force, pilot input (N)
        
Methods:
    
[nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) returns 
    nu[k+1] and u_actual[k+1] using Euler's method. The control inputs are:

    u_control = [ n1 n2 ]' where 
        n1: propeller shaft speed, left (rad/s)
        n2: propeller shaft speed, right (rad/s)

u = headingAutopilot(eta,nu,sampleTime) 
    PID controller for automatic heading control based on pole placement.

u = stepInput(t) generates propeller step inputs.

[n1, n2] = controlAllocation(tau_X, tau_N)     
    Control allocation algorithm.
    
References: 
  T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
     Control. 2nd. Edition, Wiley. 
     URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
from python_vehicle_simulator.lib.control import PIDpolePlacement, PIpolePlacement
from python_vehicle_simulator.lib.gnc import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat

# Class Vehicle
class blueboat:
    """
    blueboat()                                           Propeller step inputs
    blueboat('headingAutopilot',psi_d,V_c,beta_c,tau_X)  Heading autopilot
    
    Inputs:
        psi_d: desired heading angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)        
    """

    def __init__(
        self, 
        controlSystem="stepInput", 
        r = 0, 
        V_current = 0, 
        beta_current = 0,
        u_ref = 1 # m/s
    ):
        
        # Constants
        D2R = math.pi / 180     # deg2rad
        self.g = 9.81           # acceleration of gravity (m/s^2)
        rho = 1000              # density of water (kg/m^3)

        if controlSystem == "pathfollowingAutopilot":
            self.controlDescription = (
                "Heading autopilot, psi_d = "
                + str(r)
                + " deg"
                )
        else:
            self.controlDescription = "Step inputs for n1 and n2"
            controlSystem = "stepInput"

        self.ref = r
        self.u_ref = u_ref
        self.V_c = V_current
        self.beta_c = beta_current * D2R
        self.controlMode = controlSystem
        #self.tauX = tau_X  # surge force (N)

        # Initialize the blueboat USV model
        self.T_n = 0.1  # propeller time constants (s)
        self.L = 1.053    # length (m)
        self.B = 0.826   # beam (m)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # velocity vector
        self.u_actual = np.array([0, 0], float)  # propeller revolution states
        self.name = "blueboat USV (see 'blueboat.py' for more details)"

        self.controls = [
            "Left propeller shaft speed (rad/s)",
            "Right propeller shaft speed (rad/s)"
        ]
        self.dimU = len(self.controls)

        # Vehicle parameters
        m = 15.0                                 # mass (kg)
        self.mp = 0.0                           # Payload (kg)
        self.m_total = m + self.mp
        self.rp = np.array([0.0, 0.0, 0.0], float) # location of payload (m)
        rg = np.array([-0.035, -0.006, -0.031], float)     # CG for hull only (m)
        rg = (m * rg + self.mp * self.rp) / (m + self.mp)  # CG corrected for payload
        self.S_rg = Smtrx(rg)
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(self.rp)


        T_sway = 1.0        # time constant in sway (s)
        T_yaw = 1.0         # time constant in yaw (s)
        Umax = 4  # max forward speed (m/s) Educated guess, but can be changed later

        # Data for one pontoon
        self.B_pont = 0.128  # beam of one pontoon (m)
        y_pont = 0.296     # distance from centerline to waterline centroid (m)
        Cw_pont = 0.102/(self.B_pont*self.L)     # waterline area coefficient (-)
        Cb_pont = 0.35       # block coefficient, computed from m = 55 kg

        # Inertia dyadic, volume displacement and draft
        nabla = (m + self.mp) / rho  # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft

        # Ig matrix option one (simplified)
        #R44 = 0.4 * self.B  # radii of gyration (m)
        #R55 = 0.25 * self.L
        #R66 = 0.25 * self.L
        #Ig_CG = m * np.diag(np.array([R44 ** 2, R55 ** 2, R66 ** 2]))
        #self.Ig = Ig_CG - m * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp
        # Ig matrix option two (CAD)
        Ig_CAD = np.array([
				    [1.756,  -0.003,  0.069],
				    [-0.003,  1.191,  -0.01],
				    [0.069,  -0.01,  2.595]
				])
        self.Ig = Ig_CAD

        # Experimental propeller data including lever arms
        self.l1 = -y_pont  # lever arm, left propeller (m)
        self.l2 = y_pont  # lever arm, right propeller (m)
        # Thruster parameters
        thruster_max = 5.63*9.81 # kg*f *9.81 = N From Datasheet
        thruster_min = 2.8*9.81 # kg*f *9.81 = N From Datasheet
        #self.n_max = math.sqrt((0.5 * 24.4 * self.g) / self.k_pos)  # max. prop. rev.
        #self.n_min = -math.sqrt((0.5 * 13.6 * self.g) / self.k_neg) # min. prop. rev.
        self.n_max = 45.85  # max. prop. rev. RPS/60 = RPM From Datasheet
        self.n_min = -47.066 # min. prop. rev. RPS/60 = RPM From Datasheet
        thruster_diameter = 0.112 # m From Datasheet
        kt_pos = thruster_max/(rho*(thruster_diameter**4)*(self.n_max**2)) # propeller thrust coefficient Fossen eq.9.2
        kt_neg = thruster_min/(rho*(thruster_diameter**4)*(self.n_min**2)) # propeller thrust coefficient Fossen eq.9.2
        self.k_pos = rho*(thruster_diameter**4)*kt_pos #0.02588   # Positive Bollard, one propeller
        self.k_neg = rho*(thruster_diameter**4)*kt_neg #0.01222  # Negative Bollard, one propeller
        self.T_max = thruster_max*2 # Total max thrust two propellers 

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (m + self.mp) * np.identity(3)
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Zwdot = -1.0 * m
        Kpdot = -0.2 * self.Ig[0, 0]
        Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        #Aw_pont = Cw_pont * self.L * self.B_pont  # waterline area, one pontoon (computed)
        Aw_pont = 0.102  # waterline area, one pontoon (From CAD)
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont ** 3
            * (6 * Cw_pont ** 3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
            + 2 * Aw_pont * y_pont ** 2
        )
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla  # BM values
        BM_L = I_L / nabla
        KM_T = KB + BM_T    # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG    # GM values
        GM_L = KM_L - KG

        G33 = rho * self.g * (2 * Aw_pont)  # spring stiffness
        G44 = rho * self.g * nabla * GM_T
        G55 = rho * self.g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.1
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = math.sqrt(G33 / self.M[2, 2])
        w4 = math.sqrt(G44 / self.M[3, 3])
        w5 = math.sqrt(G55 / self.M[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -self.T_max / Umax # Max total thrust / Umax   # specified using the maximum speed
        Yv = -self.M[1, 1]  / T_sway # specified using the time constant in sway
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])
        self.Xuu = 0.5*rho*Aw_pont*Cb_pont# Nonlinear surge speed dampning

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

        # Heading autopilot
        self.e_int = 0  # integral state
        wb_h = 2.0 # wb = 1.0 -> 2.0 for heading
        self.zeta_h = 1.0 # safe behaviour 1.0, agressive behavirou 0.8
        self.wn_h = 1/(math.sqrt(1-2*self.zeta_h**2+math.sqrt(4*self.zeta_h**4-4*self.zeta_h**2+2)))*wb_h #2.5   # PID pole placement
        #print(self.wn_h)


        #Surge speed autopilot
        self.u_int = 0
        self.u_max = Umax
        self.u_d = 0
        self.u_dot_d = 0
        wb_s = 1.0 # wb = 0.5 -> 1.0 for heading
        self.zeta_s = 1.0 # safe behaviour 1.0, agressive behavirou 0.8
        self.wn_s = 1/(math.sqrt(1-2*self.zeta_s**2+math.sqrt(4*self.zeta_s**4-4*self.zeta_s**2+2)))*wb_s #2.5   # PID pole placment
        #print(self.wn_s)

        # Reference model
        self.r_max = 10 * math.pi / 180  # maximum yaw rate
        self.psi_d = 0   # angle, angular rate and angular acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = 0.5  # desired natural frequency in yaw
        self.zeta_d = 1.0  # desired relative damping ratio

        #ILOS Params
        self.look_ahead_dist = 6 #m Set to 3 to 6 times the legth of the boat
        self.kappa = 0.2 #0.1 slow, 1 is aggressive
        self.integrator = 0


    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the blueboat USV equations of motion using Euler's method.
        """

        # Input vector
        n = np.array([u_actual[0], u_actual[1]])

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge vel.
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway vel.

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

        CA = m2c(self.MA, nu_r)
        # Uncomment to cancel the Munk moment in yaw, if stability problems
        #CA[5, 0] = 0  
        #CA[5, 1] = 0 
        #CA[0, 5] = 0
        #CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(eta[3], eta[4], eta[5])
        f_payload = np.matmul(R.T, np.array([0, 0, self.mp * self.g], float))              
        m_payload = np.matmul(self.S_rp, f_payload)
        g_0 = np.array([ f_payload[0],f_payload[1],f_payload[2], 
                         m_payload[0],m_payload[1],m_payload[2] ])

        # Control forces and moments - with propeller revolution saturation
        thrust = np.zeros(2)
        for i in range(0, 2):

            n[i] = sat(n[i], self.n_min, self.n_max)  # saturation, physical limits

            if n[i] > 0:  # positive thrust
                thrust[i] = self.k_pos * n[i] * abs(n[i])
            else:  # negative thrust
                thrust[i] = self.k_neg * n[i] * abs(n[i])

        # Control forces and moments
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                0,
                0,
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1],
            ]
        )

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]
        # --- ADD THIS LINE FOR SURGE ---
        # Subtract quadratic drag force (F = -C * |u| * u)
        tau_damp[0] = tau_damp[0] - self.Xuu * 0.09 * abs(nu_r[0]) * nu_r[0] # the 0.09 factor reduces teh quadratic drac that wourl otehrwise explode Can be tuned

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)
        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            + g_0
        )

        nu_dot = Dnu_c + np.matmul(self.Minv, sum_tau)  # USV dynamics
        n_dot = (u_control - n) / self.T_n  # propeller dynamics

        # Forward Euler integration [k+1]
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        return nu, u_actual


    def controlAllocation(self, tau_X, tau_N):
        """
        [n1, n2] = controlAllocation(tau_X, tau_N)
        """
        tau = np.array([tau_X, tau_N])  # tau = B * u_alloc
        #print(tau)
        u_alloc = np.matmul(self.Binv, tau)  # u_alloc = inv(B) * tau

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))

        # 2. Check Saturation
        highest_requested = max(abs(n1), abs(n2))

        if highest_requested > self.n_max:
            # Calculate the required steering difference (turning power)
            diff = n1 - n2
            
            # If the turn ITSELF is physically impossible (e.g. diff > 200), clamp it
            if abs(diff) > 2 * self.n_max:
                diff = np.sign(diff) * 2 * self.n_max

            # 3. Apply Heading Priority
            # Shift the RPMs down so the highest one hits the ceiling,
            # but the difference between them stays exactly the same.
            if n1 > n2: # Turning Right (Port/Left motor is dominant)
                n1 = self.n_max
                n2 = self.n_max - diff
            else:       # Turning Left (Starboard/Right motor is dominant)
                n2 = self.n_max
                n1 = self.n_max + diff
        #print(n1)
        #print(n2)
        return n1, n2


    def speedController(self, nu, sampleTime):
        """
        u = speedAutopilot(eta,nu,sampleTime) is a PI controller
        for automatic speed control based on pole placement.

        tau_N = (T/K) * a_d + (1/K) * rd
               - Kp * ( ssa( psi-psi_d ) + Td * (r - r_d) + (1/Ti) * z )

        """
        u = nu[0]  # surge speed
        e_u = u - self.u_d  # surge speed tracking error
        u_target = self.u_ref # surge speed setpoint

        wn =  self.wn_s  # PID natural frequency
        zeta = self.zeta_s  # PID natural relative damping factor
        wn_d = self.wn_d  # reference model natural frequency
        zeta_d = self.zeta_d  # reference model relative damping factor

        m = self.m_total + self.MA[0,0]  # (mass-Xudot) = 15
        d = self.Xuu
        k = 0
        tau_max = self.T_max

        # Heading PID feedback controller with 3rd-order reference model
        [tau_X, self.u_int, self.u_d, self.u_dot_d] = PIpolePlacement(
            self.u_int, # integral state
            e_u,
            self.u_d,
            self.u_dot_d,
            m,
            d,
            k,
            wn_d,
            zeta_d,
            wn,
            zeta,
            u_target,
            self.u_max,
            sampleTime,
            tau_max
        )

        return tau_X


    def headingController(self, eta, nu, sampleTime):
        """
        u = headingAutopilot(eta,nu,sampleTime) is a PID controller
        for automatic heading control based on pole placement.

        tau_N = (T/K) * a_d + (1/K) * rd
               - Kp * ( ssa( psi-psi_d ) + Td * (r - r_d) + (1/Ti) * z )

        """
        psi = eta[5]  # yaw angle
        r = nu[5]  # yaw rate
        e_psi = psi - self.psi_d  # yaw angle tracking error
        e_r = r - self.r_d  # yaw rate tracking error
        psi_ref = self.ref#self.ref * math.pi / 180  # yaw angle setpoint


        wn = self.wn_h  # PID natural frequency
        zeta = self.zeta_h  # PID natural relative damping factor
        wn_d = self.wn_d  # reference model natural frequency
        zeta_d = self.zeta_d  # reference model relative damping factor

        m = self.Ig[2,2] + self.MA[5,5] #(Iz-Nrdot)/1 moment of inertia in yaw including added mass
        d = self.D[5,5] #Nr/1
        k = 0

        # Heading PID feedback controller with 3rd-order reference model
        [tau_N, self.e_int, self.psi_d, self.r_d, self.a_d] = PIDpolePlacement(
            self.e_int,
            e_psi,
            e_r,
            self.psi_d,
            self.r_d,
            self.a_d,
            m,
            d,
            k,
            wn_d,
            zeta_d,
            wn,
            zeta,
            psi_ref,
            self.r_max,
            sampleTime,
        )


        return tau_N


    
    def getDesiredheading(self, x, y, wp_prev, wp_next,sampleTime):
        """
        Computes the desired heading to follow the line between wp_prev and wp_next.
        
        Args:
            x, y: Current USV position (NED frame)
            wp_prev: Previous waypoint [x, y]
            wp_next: Target waypoint [x, y]
            
        Returns:
            psi_d (float): Desired heading in Radians.
        """
        # 1. Geometry of the path
        x_prev, y_prev = wp_prev
        x_next, y_next = wp_next
        
        # Path angle (alpha_k) relative to North
        alpha_k = np.arctan2(y_next - y_prev, x_next - x_prev)
        
        # 2. Cross-track error (ye)
        # Formula: -(x - x_prev)*sin(alpha) + (y - y_prev)*cos(alpha)
        # Positive ye means we are to the right of the path (need to turn left)
        ye = -(x - x_prev) * np.sin(alpha_k) + (y - y_prev) * np.cos(alpha_k)

        # PI gains based on eq.12.108
        kp = 1/self.look_ahead_dist
        ki = self.kappa*kp
        
        # Compute Desired Heading using PI
        psi_d = alpha_k - np.arctan((kp*ye + ki*self.integrator))

        # Integrator_dot eq.12.109
        integrator_dot = (self.look_ahead_dist*ye) / (  (self.look_ahead_dist)**2 + (ye + self.kappa*self.integrator)**2)

        # Integral error
        self.integrator += integrator_dot * sampleTime
        
        # Normalize angle to [-pi, pi]
        psi_d = (psi_d + np.pi) % (2 * np.pi) - np.pi
        
        return psi_d, ye

    def speedHeadingAutopilot(self, eta, nu, sampleTime):
        tau_N = self.headingController(eta, nu, sampleTime)
        tau_X = self.speedController(nu, sampleTime)
        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)
        return u_control
    
    def pathfollowingAutopilot(self,eta,nu,sampleTime,t):

        # 1. Calculate Sine Wave Heading
        # Formula: Bias + Amplitude * sin(omega * t)
        #psi_d = np.sin(t/20)
        #print(psi_d*(180/math.pi))
        #print(t)
        # 2. (Optional) Normalize if you cross +/- PI (not strictly needed for small angles)
        #self.ref = (psi_d + np.pi) % (2 * np.pi) - np.pi

        wp_prev = [3,0]
        wp_next = [3,100]
        self.ref, _ = self.getDesiredheading(eta[0], eta[1], wp_prev, wp_next, sampleTime)
        print(self.ref*(180/math.pi))

        tau_N = self.headingController(eta, nu, sampleTime)
        tau_X = self.speedController(nu, sampleTime)
        [n1, n2] = self.controlAllocation(tau_X, tau_N)
        u_control = np.array([n1, n2], float)
        return u_control
        



    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs.
        """
        n1 = 100  # rad/s
        n2 = 80

        if t > 30 and t < 100:
            n1 = 80
            n2 = 120
        else:
            n1 = 0
            n2 = 0

        u_control = np.array([n1, n2], float)

        return u_control
