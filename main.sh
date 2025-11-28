#!/bin/bash

echo "==============================="
echo "  CONTROL SIMULATION LAUNCHER"
echo "==============================="
echo
echo "1 - Open Spring–Damper System "
echo "2 - Run Inverted Pendulum Simulation"
echo "3 - Exit"
echo
read -p "Enter choice: " choice

case $choice in
  1)
    echo "Opening Spring–Damper System Notebook..."
    jupyter notebook spring_damper_system.ipynb
    ;;
  2)
    echo "Running Inverted Pendulum Simulation..."
    python3 invertedpendulum.py
    ;;
  3)
    exit 0
    ;;
  *)
    echo "Invalid choice."
    ;;
esac
