import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    x = [path[0][0]]
    y = [path[0][1]]
    t = [0]
    d_est = 0.0
    for i in range(1,len(path)):
        x.append(path[i][0])
        y.append(path[i][1])
        d_est = d_est+np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)
        t.append(d_est/V_des)
    x_spl = scipy.interpolate.splrep(t,x,s=2)
    y_spl = scipy.interpolate.splrep(t,y,s=2)
    
    t_smoothed = np.arange(0,t[-1],dt)
    x_smoothed = scipy.interpolate.splev(t_smoothed,x_spl)
    y_smoothed = scipy.interpolate.splev(t_smoothed,y_spl)
    xd_smoothed = scipy.interpolate.splev(t_smoothed,x_spl,der = 1)
    yd_smoothed = scipy.interpolate.splev(t_smoothed,y_spl,der = 1)
    th_smoothed = np.arctan2(yd_smoothed,xd_smoothed)
    xdd_smoothed = scipy.interpolate.splev(t_smoothed,x_spl,der = 2)
    ydd_smoothed = scipy.interpolate.splev(t_smoothed,y_spl,der = 2)
    
    traj_smoothed = np.transpose(np.vstack((x_smoothed,y_smoothed,th_smoothed,xd_smoothed,yd_smoothed,xdd_smoothed,ydd_smoothed)))
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
