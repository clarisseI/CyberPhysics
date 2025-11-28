# Control Systems Simulation

Python implementations of classical control systems with numerical simulation and state estimation.

## 📋 Overview

This repository contains two control system simulations:

1. **Mass-Spring-Damper System** - Euler method comparison of linear vs nonlinear damping
2. **Inverted Pendulum** - LQR control with Luenberger observer design

## 📁 Project Structure

```
project/
├── mass_spring_damper.py    # Spring-damper Euler simulation
├── inverted_pendulum.py     # LQR + observer pendulum control
├── main.sh                  # Launcher script
└── README.md
```

## 🛠️ Prerequisites

- Python 3.6+
- NumPy
- Matplotlib
- Python Control Systems Library

Install dependencies:
```bash
pip install numpy matplotlib control
```

## 🚀 Usage

### Option 1: Launcher Script (Recommended)

```bash
./main.sh
```

Select simulation:
- `1` - Mass-Spring-Damper System
- `2` - Inverted Pendulum with LQR & Observer
- `3` - Exit

### Option 2: Run Individually

**Mass-Spring-Damper:**
```bash
jupyter notebook spring_damper_system.ipynb
```

**Inverted Pendulum:**
```bash
python3 inverted_pendulum.py
```

## 📊 Simulations

### 1. Mass-Spring-Damper System

**System Equations:**
- Linear: ẍ + (c/m)ẋ + (k/m)x = 0
- Nonlinear: ẍ + (c/m)|ẋ|ẋ + (k/m)x = 0

**Features:**
- Compares linear vs nonlinear damping
- Tests multiple time steps (0.1s, 0.01s, 0.005s)
- Analyzes Euler method accuracy

**Parameters:**
- Mass (m) = 1.0 kg
- Spring constant (k) = 2 N/m
- Damping coefficient (c) = 0.5 N·s/m
- Initial position = 1.0 m

**Output Plots:**
- Position comparison for each time step
- Time step convergence analysis

### 2. Inverted Pendulum with LQR & Observer

**System:**
- Cart-pendulum system with friction
- Linearized dynamics around upright position
- Discrete-time LQR controller
- Luenberger observer for state estimation

**Features:**
- Full-state feedback control
- Observer-based control comparison
- Performance metrics (overshoot, settling time, control effort)
- Estimation error analysis

**Parameters:**
- Pendulum mass (m) = 0.16 kg
- Cart mass (M) = 2.3 kg
- Pendulum length (L) = 0.2 m
- Sample time (Ts) = 0.01 s
- Initial angle = 0.2 rad (~11.5°)

**Output Plots:**
- Pendulum angle θ(t)
- Cart position y(t)
- Control input u(t)
- State estimation errors

**Metrics:**
- Peak and RMS control effort
- Overshoot for angle and position
- Settling time (within 2° and 2cm)

## 🎯 Key Concepts

### Euler Method
Forward Euler numerical integration for solving ODEs:
```
x[k+1] = x[k] + ẋ[k] · Δt
```

### LQR Control
Linear Quadratic Regulator minimizes cost function:
```
J = ∫(x'Qx + u'Ru)dt
```
Controller: u = -Kx

### Luenberger Observer
State estimator with observer gain L:
```
x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
```

## 📈 Expected Results

**Mass-Spring-Damper:**
- Nonlinear damping shows faster decay
- Smaller time steps improve accuracy
- Convergence visible across time step comparison

**Inverted Pendulum:**
- Both controllers stabilize pendulum to upright position
- Observer-based control closely matches full-state feedback
- Estimation errors converge to zero
- Control effort comparable between methods

## 📄 License

MIT License