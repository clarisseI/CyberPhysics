import numpy as np
import threading
import time
import scipy.io as sci
import matplotlib.pyplot as plt


# ------------------------ Dynamics and Control ------------------------

def generate_waypoints_figure_8(x_amplitude, y_amplitude, omega, z0, steps):
    """Generate figure-8 waypoints in 3D space."""
    t = np.linspace(0, 4 * np.pi / omega, steps)
    x = x_amplitude * np.sin(omega * t)
    y = y_amplitude * np.sin(2 * omega * t)
    z = np.full_like(t, z0)
    waypoints = np.zeros((steps, 12))
    for i in range(steps):
        waypoints[i, :] = [0, x[i], 0, y[i], 0, z[i], 0, 0, 0, 0, 0, 0]
    return waypoints


def load_matrix_K(folder):
    return sci.loadmat(folder)['K']


# ------------------------ Shared Global Variables ------------------------

x0 = np.zeros(12) # Current state
control_u = np.zeros(4) # Control input
x_sp = np.zeros(12)  # Target waypoint
noise_x0 = np.zeros(12)  # Noisy state for sensor
stopEvent = threading.Event()


# ------------------------ Threaded Modules ------------------------

def simulator_thread(A, B, M, g, Ts, x_traj, y_traj, z_traj, sim_t):
    """Simulates quadrotor plant dynamics using the control input."""
    global x0, control_u
    prev_control = np.zeros(4)
    start_time = time.time()
    
    while not stopEvent.is_set():
        try:
            try:
                # Read current control input
                cu = control_u.copy()
                prev_control = cu.copy()
            except Exception as e:
                # Use previous control input if error occurs
                print(f"[SIMULATOR] Error reading control input: {e}")
                cu = prev_control.copy()
                cu[0]= M * g  # maintain hover if no control available
            # Simulate dynamics
            dx = A @ x0 + B @ cu
            # Update state
            x0 = x0 + dx * Ts

            # Store trajectory
            sim_t.append(time.time()-start_time)
            x_traj.append(x0[1])
            y_traj.append(x0[3])
            z_traj.append(x0[5])
        except Exception as e:
            print(f"[SIMULATOR] Error in main loop: {e}")
        time.sleep(Ts)
        
def sensor_thread(Ts, x_sensor, y_sensor, z_sensor, sens_t, noise_std='None'):
    """Reads simulator state ( add noise or delay)."""
    global x0, noise_x0

    prev_x0 = np.zeros(12)
    start_time = time.time()
    
    while not stopEvent.is_set():
        try:
            try:
                # Read current state from simulator
                noise_x0 = x0.copy()
                prev_x0 = noise_x0.copy()
            except Exception as e:
                # Use previous state if error occurs
                print(f"[SENSOR] Error reading simulator state: {e}")
                noise_x0 = prev_x0.copy()
            if noise_std == 'delay':
                time.sleep(1)  # simulate delay
            elif noise_std == 'noise':
                noise = np.random.normal(0, 0.01, size=12)
                noise_x0 += noise
                
            # Update sensor readings
            sens_t.append(time.time()-start_time)
            x_sensor.append(noise_x0[1])
            y_sensor.append(noise_x0[3])
            z_sensor.append(noise_x0[5])

        except Exception as e:
            print(f"[SENSOR] Error in main loop: {e}")
        time.sleep(Ts)
        
def ground_thread(waypoints, Ts, x_ground, y_ground, z_ground, gnd_t):
    """Updates target waypoint based on distance to current position."""
    global noise_x0, x_sp

    y = 0
    prev_x0 = np.zeros(12)
    start_time = time.time()
    
    while not stopEvent.is_set():
        try:
            try:
                # Read current state from sensor
                x = noise_x0.copy()
                prev_x0 = x.copy()
            except Exception as e:
                # Use previous state if error occurs
                x = prev_x0.copy() 
                print(f"[GROUND] Error reading sensor state: {e}")
                
            dist = np.linalg.norm(x[[1, 3, 5]] - waypoints[y, [1, 3, 5]])
           
            if dist <= 0.05 and y < len(waypoints) - 1:
                y += 1
                print(f"[GROUND] Reached waypoint {y}/{len(waypoints)} at time {time.time() - start_time:.2f}s")
                
            
            # Update target waypoint
            try:
                x_sp = waypoints[y].copy()

                gnd_t.append(time.time()-start_time)
                x_ground.append(x_sp[1])
                y_ground.append(x_sp[3])
                z_ground.append(x_sp[5])
            except Exception as e:
                print(f"[GROUND] Error setting target: {e}")
                
        except Exception as e:
            print(f"[GROUND] Error in main loop: {e}")
        
        time.sleep(Ts)

def controller_thread(K, Ts, force, phi, theta, psi, ctrl_time, e_x, e_y, e_z, e_time):
    """Computes control input using LQR based on current state and target waypoint."""
    global noise_x0, control_u, x_sp

    prev_x0 = np.zeros(12)
    prev_x_sp = np.zeros(12)
    start_time = time.time()
    
    while not stopEvent.is_set():
        try:
            # Read current state
            try:
                x = noise_x0.copy()
                prev_x0 = x.copy()
            except Exception as e:
                print(f"[CONTROLLER] Error reading sensor state: {e}")
                x = prev_x0.copy()
            # Read target waypoint
            try:
                x_sp = x_sp.copy()
                prev_x_sp = x_sp.copy()
            except Exception as e:
                print(f"[CONTROLLER] Error reading target waypoint: {e}")
                x_sp = prev_x_sp.copy()
            
            # LQR Control Law
            e=x - x_sp
            cu = -K @ (e)

            force.append(cu[0])
            phi.append(cu[1])
            theta.append(cu[2])
            psi.append(cu[3])
            e_x.append(e[0])
            e_y.append(e[1])
            e_z.append(e[2])
            e_time.append(time.time() - start_time)
            ctrl_time.append(time.time() - start_time)
            
            # Send control
            try:
                control_u = cu.copy()
                
            except Exception as e:
                print(f"[CONTROLLER] Error setting control: {e}")
            
        except Exception as e:
            print(f"[CONTROLLER] Error in main loop: {e}")
        
        time.sleep(Ts)


# ------------------------ Main Simulation ------------------------

if __name__ == "__main__":
    x_amp, y_amp, z0 = 5, 5, 2
    T = 20
    omega = 2 * np.pi / T
    steps = 100
    Ts = 0.001
    sim_time = 100
    
    # Quadrotor Parameters
    M = 0.6  # mass (Kg)
    L = 0.2159 / 2  # arm length (m)
    g = 9.81  # acceleration due to gravity (m/s^2)
    m = 0.410  # Sphere mass (Kg)
    R = 0.0503513  # Sphere radius (m)
    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor + propeller mass (Kg)

    # Inertia
    Jx = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jy = (2 * m * R) / 5 + 2 * L ** 2 * m_m
    Jz = (2 * m * R) / 5 + 4 * L ** 2 * m_m

    # Linearized Model in Hovering Mode
    # Define the A matrix
    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    # Define the B matrix
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [-1 / M, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1 / Jx, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1 / Jy, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1 / Jz],
                  [0, 0, 0, 0]])
    # Generate waypoints
    waypoints = generate_waypoints_figure_8(x_amp, y_amp, omega, z0, steps)
    print(f"[MAIN] Generated {len(waypoints)} waypoints for the figure-8 trajectory.")
    
    K = load_matrix_K("../mat/K.mat")
    x_traj, y_traj, z_traj = [], [], []
    x_sens, y_sens, z_sens = [], [], []
    x_gnd, y_gnd, z_gnd = [], [], []
    sim_t, sens_t, ctrl_t, gnd_t = [], [], [], []
    force, phi, theta, psi, ctrl_time = [], [], [], [], []
    e_x, e_y, e_z, e_time = [], [], [], []
    
    # Start threads
    threads = [
        threading.Thread(target=simulator_thread, args=(A, B, M, g, 0.001, x_traj, y_traj, z_traj, sim_t), name="Simulator"),
        threading.Thread(target=sensor_thread, args=(0.003, x_sens, y_sens, z_sens, sens_t,'noise'), name="Sensor"),
        threading.Thread(target=controller_thread, args=(K, 0.001, force, phi, theta, psi, ctrl_time, e_x, e_y, e_z, e_time), name="Controller"),
        threading.Thread(target=ground_thread, args=(waypoints, 0.001, x_gnd, y_gnd, z_gnd, gnd_t), name="Ground")
    ]

    print("Starting threads...\n")
    for t in threads: 
        t.start()
    
    try:
        time.sleep(sim_time)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("\nStopping simulation...")
        stopEvent.set()
        for t in threads: 
            t.join(timeout=2)

    print(f"\n[MAIN] Simulation complete. Collected {len(x_traj)} trajectory points.")

    # ------------------------figure 8 simulation  ------------------------
    if len(x_traj) > 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot actual trajectory and reference waypoints
        ax.plot(x_traj, y_traj, z_traj, color="red", linewidth=2, label="Drone trajectory")
        ax.plot(waypoints[:, 1], waypoints[:, 3], waypoints[:, 5], "--", color="blue", linewidth=1, label="Waypoints")

        # Mark Start and End points
        ax.scatter(x_traj[0], y_traj[0], z_traj[0], color="green", s=60, marker="o")
        ax.text(x_traj[0], y_traj[0], z_traj[0] + 0.3, "Start", color="green", fontsize=10)

        ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color="black", s=60, marker="X")
        ax.text(x_traj[-1], y_traj[-1], z_traj[-1] + 0.3, "Finish", color="black", fontsize=10)

        # Axis labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(" Figure-8 Trajectory Tracking")
        ax.legend()
        plt.show()
    else:
        print("[MAIN] No trajectory data collected!")
        
# ------------------------ X position over time------------------------
    plt.figure()
    if len(sim_t) > 0: plt.plot(sim_t, x_traj, 'r-', label='Simulator')
    if len(sens_t) > 0: plt.plot(sens_t, x_sens, 'g:', label='Sensor')
    if len(gnd_t) > 0: plt.plot(gnd_t, x_gnd, 'b--', label='Ground')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# ------------------------ Y position over time------------------------
    plt.figure()
    if len(sim_t) > 0: plt.plot(sim_t, y_traj, 'r-', label='Simulator')
    if len(sens_t) > 0: plt.plot(sens_t, y_sens, 'g:', label='Sensor')
    if len(gnd_t) > 0: plt.plot(gnd_t, y_gnd, 'b--', label='Ground')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# ------------------------ Z position over time------------------------
    plt.figure()
    if len(sim_t) > 0: plt.plot(sim_t, z_traj, 'r-', label='Simulator')
    if len(sens_t) > 0: plt.plot(sens_t, z_sens, 'g:', label='Sensor')
    if len(gnd_t) > 0: plt.plot(gnd_t, z_gnd, 'b--', label='Ground')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title('Z Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# ------------------------ Control Inputs over time------------------------
    plt.figure()
    if len(ctrl_time) > 0: plt.plot(ctrl_time, force, 'r-', label='Force')
    if len(ctrl_time) > 0: plt.plot(ctrl_time, phi, 'g--', label='Phi')
    if len(ctrl_time) > 0: plt.plot(ctrl_time, theta, 'b:', label='Theta')
    if len(ctrl_time) > 0: plt.plot(ctrl_time, psi, 'm-.', label='Psi')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Inputs')
    plt.title('Control Inputs vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# ------------------------ Errors over time ------------------------
    plt.figure()
    if len(e_time) > 0:
        plt.plot(e_time, e_x, 'r-', label='Error X')
        plt.plot(e_time, e_y, 'g--', label='Error Y')
        plt.plot(e_time, e_z, 'b:', label='Error Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Tracking Errors vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
