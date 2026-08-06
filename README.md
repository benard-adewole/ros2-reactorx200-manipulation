# ReactorX-200 5-DoF Autonomous Manipulation & Launch Pipeline

**Notice on Repository History:**  
*This repository serves as a centralized archive for autonomous robotic manipulation pipelines originally developed and benchmarked between **March 2025 and April 2025**. The relative commit timestamps reflect the date of this archival migration, while the underlying architecture and hardware executions correspond to the active project timeline detailed below.*

---

## Project Overview
**Sequential Operational Sequence:** Demonstration of a custom 5-Degree-of-Freedom (5-DoF) robotic arm executing an autonomous operational pipeline to pick, load, and dynamically launch a ball into a target basket, utilizing vision-based target acquisition, inverse kinematics, and ROS 2/Python control architectures.

---

## Core Code Architecture

### 1. Perception & Computer Vision (`camera.py`)
* Implements the `Camera` class for the RealSense camera.
* Handles frame capture and color space conversions.
* Manages camera calibration data and AprilTag tracking for spatial localization.
* Computes world-to-camera and camera-to-world coordinate transformations.

### 2. Kinematics & Mathematical Modeling (`kinematics.py`, `config/`)
* **Forward & Inverse Kinematics:** Implements mathematical models for the RX200 arm.
* **Denavit-Hartenberg (DH) & Product of Exponentials (PoX):** Utilizes `rx200_dh.csv` and `rx200_pox.csv` parameter matrices to calculate precise joint configurations and spatial trajectories.

### 3. Hardware Abstraction & Control (`rxarm.py`, `control_station.py`)
* **RXArm Class:** Interfaces directly with Dynamixel servos for real-time joint feedback and velocity/position command distribution.
* **Control Station:** Manages multithreaded execution and callback loops, coordinating the state machine with kinematics solver flags.

### 4. State Machine & Execution (`state_machine.py`)
* Acts as the core operational controller, managing state transitions for target detection, payload acquisition, pre-grasp alignment, and high-velocity dynamic launching.

---

## Core Technologies & Frameworks
* **Languages:** Python, C++
* **Frameworks & Middleware:** ROS 2, Interbotix SDK
* **Algorithmic Concepts:** Inverse Kinematics, Forward Kinematics, Product of Exponentials (PoX), Denavit-Hartenberg (DH) Parameters, AprilTag Spatial Tracking, Dynamic Trajectory Planning

---

## Author
**Benard Adewole**  
Robotics Software Engineer | M.S. Robotics, University of Michigan
