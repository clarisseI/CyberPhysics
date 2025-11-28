#-------------------------Luenberger observer------------------------------------#

import numpy as np
import control as ct
import matplotlib.pyplot as plt


#-----------------------Parameters of inverted pendulum system---------#
m=0.16   #mass of the pendulum ([kg]
M=2.3    #mass of the cart [kg]
b=0.83   #friction coefficient of the cart [N*s/m]
k=0.002  #friction coefficient of the pendulum [N*s*m/R]
g=9.81   #gravity acceleration [m/s^2]
L= 0.2   #This is the length from the pin to the bob [m]

#-----------------------Linearized model for LQR design ---------#
A= np.array([
    [0,1,0,0],
    [(M+m)*g/(L*M),-(M+m)*(k* L/m)/(L*M),0,b/(L*M)],
    [0,0,0,1],
    [-g * (m/M), (k/m)/M, 0, -b/M]
])
print("A matrix:\n", A)
B=np.array([[0],
            [-1/(M * L)],  
            [0], 
            [1/M]          
])
print("B matrix:\n", B)
C= np.array([
    [1,0,0,0],
    [0,0,1,0]
])
print("C matrix:\n", C)

D= np.zeros((2,1))
print("D matrix:\n", D)
n=A.shape[0]  #number of states 
p=B.shape[1]  #number of inputs
q=C.shape[0]  #number of outputs

#-------------------------Discretization -----------------------
Ts=0.01  
sys=ct.ss(A,B,C,D)
sysd= ct.c2d(sys,Ts)
Ad, Bd, Cd, Dd = sysd.A, sysd.B, sysd.C, sysd.D

print("Shape Ad:", Ad.shape)
print("Shape Bd:", Bd.shape)
print("Shape Cd:", Cd.shape)

#-----------------------LQR design ----------------------------#
Q=np.diag([1200,1500,10,1])
R=np.array([[0.02]])
[K,_,_]= ct.dlqr(Ad, Bd, Q, R)
print("State feedback gain K:", K.shape)

#--------------Observer design----------------------
Q0=np.eye(n)
R0=np.eye(2)
[L_temp,_,_]=ct.dlqr(Ad.T, Cd.T, Q0, R0)
L = L_temp.T  # Transpose to get correct shape (4, 2)
print("Observer gain L:", L.shape)

#-----------------------Simulation parameters ----------------------------# 
T=20
tt= np.arange(0, T+Ts, Ts)
Ns= tt.size
bb=10

#initial condition: pendulum angle non-zero, cart at rest
x0 = np.array([[0.2], [0], [0], [0]]) #theta=0.2 rad (~11.46 deg)
hat_x0 = np.zeros((4, 1))  # Initial state estimate (all zeros)


print("Initial state x0:", x0.shape)
print("Initial state estimate hat_x0:", hat_x0.shape)


def full_state_feedback_control_simulation(A, B, K, x0, Ns, Ts):
    """
    Full state feedback control simulation.
    Returns:
        State and control input trajectories.
    """
    x = x0.copy()
    xx, uu = [x.copy()], []
    
    for _ in range(Ns-1):
        u=-K @ x
     
        # State update
        x_dot = A @ x + B @ u
        x= x + x_dot * Ts
        xx.append(x.copy())
        uu.append(u.item())

    return np.array(xx), np.array(uu)


def observer_based_state_feedback_control_simulation(Ad, Bd, Cd, K, L, x0, hat_x0, Ns, bb):    
    """
    Observer based state feedback control simulation.
    Returns:
        True state, estimated state, and control input trajectories.
    """
    x = x0.copy()          
    x_hat= hat_x0.copy() 
    hat_xx, uu = [x_hat.copy()], []

    for _ in range(Ns-1):
        y = Cd @ x  
        u = -K @ x_hat  
        x_hat = Ad @ x_hat + Bd @ u + L @ (y - Cd @ x_hat)
        x = Ad @ x + Bd @ u
        hat_xx.append(x_hat.copy())
        uu.append(u.item())

    return np.array(hat_xx), np.array(uu)

def overshoot(x, desired_theta=0.0, desired_y=0.0):
    """
    Calculate the percentage overshoot of a response.
    Args:
        x: Response array.
        desired_theta: The desired steady-state value for the pendulum angle.
        desired_y: The desired steady-state value for the cart position.
     """
    theta = x[:, 0] * 180.0 / np.pi
    y = x[:, 2]
    theta_overshoot = np.max(np.abs(theta - desired_theta))
    y_overshoot = np.max(np.abs(y - desired_y))
    return float(theta_overshoot), float(y_overshoot)

def settling_time(x_traj, tt, tol_theta_deg=2.0, tol_y=0.02):
    """
    Calculate the settling time of a response.
    Args:
        x_traj: State trajectory.
        tt: Time vector.
        tol_theta_deg: Tolerance for pendulum angle.
        tol_y: Tolerance for cart position.
    """
    theta_deg = np.abs(x_traj[:, 0] * 180.0 / np.pi)
    y_abs = np.abs(x_traj[:, 2])
    for i in range(len(tt)):
        if np.all(theta_deg[i:] < tol_theta_deg) and np.all(y_abs[i:] < tol_y):
            return float(tt[i])
    return None

# ----------------------- pendulum angle --------------------
def plot_theta(tt, xx_full, xx_obs):
    plt.figure(figsize=(8, 3.5))
    plt.plot(tt, xx_full[:, 0] * 180 / np.pi, 'b-', label='θ (Full-state)')
    plt.plot(tt, xx_obs[:, 0] * 180 / np.pi, 'r--', label='θ (Observer-based)')
    plt.xlabel('Time [s]');
    plt.ylabel('Pendulum Angle [deg]')
    plt.title('Pendulum Angle θ(t)'); 
    plt.grid(True); plt.legend(); 
    plt.tight_layout()
    plt.show()
# ----------------------- cart position --------------------   
def plot_position(tt, xx_full, xx_obs):
    plt.figure(figsize=(8, 3.5))
    plt.plot(tt, xx_full[:, 2], 'b-', label='y (Full-state)')
    plt.plot(tt, xx_obs[:, 2], 'r--', label='y (Observer-based)')
    plt.xlabel('Time [s]'); 
    plt.ylabel('Cart Position [m]')
    plt.title('Cart Position y(t)'); 
    plt.grid(True); plt.legend(); 
    plt.tight_layout()
    plt.show()
# ----------------------- Control input plot --------------------  
def plot_control(tt, uu_full, uu_obs):
    t_u = tt[:-1]
    plt.figure(figsize=(8, 3.5))
    plt.plot(t_u, uu_full, 'b-', label='u (Full-state)')
    plt.plot(t_u, uu_obs, 'r--', label='u (Observer-based)')
    plt.xlabel('Time [s]'); plt.ylabel('Control Input [N]')
    plt.title('Control Input u(t)'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()
# ----------------------- Estimation error plot --------------------   
def plot_est_errors(tt, est_errors):
    plt.figure(figsize=(8, 3.5))
    plt.plot(tt, est_errors[:, 0] * 180 / np.pi, label='θ error [deg]')
    plt.plot(tt, est_errors[:, 1] * 180 / np.pi, label='θ̇ error [deg/s]')
    plt.plot(tt, est_errors[:, 2], label='y error [m]')
    plt.plot(tt, est_errors[:, 3], label='ẏ error [m/s]')
    plt.xlabel('Time [s]'); 
    plt.ylabel('Estimation Error'); 
    plt.title('Full State vs Observer State Estimation Errors')
    plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("INVERTED PENDULUM SIMULATION WITH LQR AND OBSERVER")
    print("="*60 + "\n")

    xx_full, uu_full = full_state_feedback_control_simulation(A, B, K, x0, Ns, Ts)
    hat_xx_obs, uu_obs = observer_based_state_feedback_control_simulation(
        Ad, Bd, Cd, K, L, x0, hat_x0, Ns, bb
    )
    estimation_errors = xx_full - hat_xx_obs

   
    print("Final States:")
    print(f" Final θ full-state: {(xx_full[-1, 0]*180/np.pi).item():.4f} deg")
    print(f" Final y full-state: {(xx_full[-1, 2]).item():.4f} m")
    print(f" Final θ observer-based: {(hat_xx_obs[-1, 0]*180/np.pi).item():.4f} deg")
    print(f" Final y observer-based: {(hat_xx_obs[-1, 2]).item():.4f} m")

    print(f"\nControl Effort Metrics:")
    print(f"Full-state controller: Peak = {(np.max(np.abs(uu_full))).item():.4f} N, RMS = {(np.sqrt(np.mean(uu_full**2))).item():.4f} N")
    print(f"Observer-based controller: Peak = {(np.max(np.abs(uu_obs))).item():.4f} N, RMS = {(np.sqrt(np.mean(uu_obs**2))).item():.4f} N")
 

    theta_ob, y_ob = overshoot(hat_xx_obs)
    st_observer = settling_time(hat_xx_obs, tt)
    theta_full, y_full = overshoot(xx_full)
    st_full = settling_time(xx_full, tt)

    print("Additional Metrics:")
    print(f"  Overshoot (θ, y) observer-based: {theta_ob:.6f} deg, {y_ob:.6f} m")
    print(f"  Overshoot (θ, y) full-state: {theta_full:.6f} deg, {y_full:.6f} m")
    print(f"  Settling time observer-based: {st_observer if st_observer is not None else 'not settled'} s")
    print(f"  Settling time full-state: {st_full if st_full is not None else 'not settled'} s")

    # ----------------------- Plots --------------------
    plot_theta(tt, xx_full, hat_xx_obs)
    plot_position(tt, xx_full, hat_xx_obs)
    plot_control(tt, uu_full, uu_obs)
    plot_est_errors(tt, estimation_errors)