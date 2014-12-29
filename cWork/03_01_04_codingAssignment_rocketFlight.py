from __future__ import division
from math import sin, cos, log, ceil, pi
import numpy

# model parameters:
m_s = 50 # [kg] the weight of the rocket shell
g = 9.81      # [m/s^2] gravitational acceleration
rho = 1.091     # [kg/m^3] is the average air density (assumed constant throughout flight)
r = 0.5       # [m] the maximum cross sectional radius of the rocket
A = pi*r**2   # [m^2] the maximum cross sectional area of the rocket
v_e = 325.0     # [m/s] the exhaust speed
C_D = 0.15    # [] the drag coefficient
m_p0 = 100.0    # [kg] at time t=0 is the initial weight of the rocket propellant

### set initial conditions ###
h0 = 0     # start at zero altitude
v0 = 0     # start at rest


def f(u):
    """Returns the right-hand side of the rocket flight system of equations.
    
    u = [h,v]
    u' = [v,-g + mDot_p*v_e/(m_s+m_p) - 0.5*rho*v**2*A*C_D/(m_s+m_p) ]
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    
    #h = u[0] # Not used anywhere
    v = u[1]
    
    return numpy.array([v,-g + mDot_p*v_e/(m_s+m_p) - 0.5*rho*v*abs(v)*A*C_D/(m_s+m_p) ]) # ohh abs(v) is sooo much important, for downward velocity, the drag must be up!



def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)





T = 100.0                          # final time
dt = 0.1                           # time increment
N = int(T/dt) + 1                  # number of time-steps
t = numpy.linspace(0.0, T, N)      # time discretization

# initialize the array containing the solution for each time-step
u = numpy.empty((N, 2))
m_p_ = numpy.empty(N)
m_p_[0] = m_p0

u[0] = numpy.array([h0, v0])# fill 1st element with initial values

# time loop - Euler method
for n in range(N-1):
    
    if t[n]>=0 and t[n]<5.0:
        mDot_p = 20.0 # kg/s
        m_p = m_p_[n] - mDot_p*dt
        #m_p = m_p0 - mDot_p*n*dt
    else:
        mDot_p = 0
        m_p = m_p_[n]
    
    
    unp1 = euler_step(u[n], f, dt)
    
    if unp1[0] < 0:
        break
    
    u[n+1] = unp1
    m_p_[n+1] = m_p
    
t = t[:n+1]
h = u[:n+1,0]
v = u[:n+1,1]

#plot(t,h)


# REMAINING FUEL  (5 points possible)
# --------------------------------------------------------------------------------------------------

idx_32s = numpy.where(t==3.2)[0][0]
print m_p_[idx_32s] # 36.0

# MAXIMUM VELOCITY  (10 points possible)
# --------------------------------------------------------------------------------------------------

print v.max()  # 235.89718920415186


print t[v.argmax()] # 5.0 s

print h[v.argmax()] # 533.054222187


# MAXIMUM HEIGHT  (10 points possible)
# --------------------------------------------------------------------------------------------------
print h.max()  # 1354.63079286

print t[h.argmax()] # 15.8 s

# IMPACT  (10 points possible)
# --------------------------------------------------------------------------------------------------
print t[-1] # 37.3
print v[-1] # -86.0531714788
