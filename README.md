# Deep Reinforcement Learning for Active Janus Nanoparticle Navigation

A physics-informed Deep Reinforcement Learning (DRL) framework for autonomous navigation of active Janus nanoparticles in stochastic environments.

This project investigates whether a Double Deep Q-Network (Double DQN) agent can learn a temporal laser ON/OFF control strategy for navigating a Brownian Janus nanoparticle toward a predefined target.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Scientific Motivation](#-scientific-motivation)
- [Physics Model](#-physics-model)
- [RL Environment Design](#-rl-environment-design)
- [DQN Architecture](#-dqn-architecture)
- [Training](#-training)
- [Baseline Strategies](#-baseline-strategies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Code Structure](#-code-structure)
- [Results](#-results)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Citation & References](#-citation--references)
- [License](#-license)

---

## Overview

Controlling active matter at the nanoscale is challenging because the dynamics are strongly affected by thermal fluctuations.

In this project, a simplified Janus nanoparticle is modeled as an active Brownian particle whose propulsion can be controlled by switching a laser ON or OFF.

A reinforcement learning agent observes the particle's state and decides whether propulsion should be activated.

The objective is to reach a predefined target while minimizing unnecessary propulsion.

The project combines:

- Stochastic particle dynamics
- Active matter
- Brownian motion
- Deep reinforcement learning
- Double DQN
- Physics-informed reward design
- Numerical simulation

> This repository is primarily a computational physics and reinforcement-learning research project developed for learning, experimentation, and model development.

---

## Scientific Motivation

At the nanoscale, thermal fluctuations play an important role in particle motion.

The Péclet number is used to characterize the relative importance of propulsion and diffusive motion:

$$
Pe = \frac{v}{\sqrt{D_T D_R}}
$$

where:

- $v$ is the propulsion velocity
- $D_T$ is the translational diffusion coefficient
- $D_R$ is the rotational diffusion coefficient

For the parameters used in this project:

$$
Pe \approx 0.30
$$

At this regime, rotational Brownian motion can significantly change the particle orientation.

Therefore, continuously activating propulsion is not necessarily optimal.

The reinforcement learning problem investigates whether an agent can learn a temporal strategy in which it sometimes waits for a favorable orientation before activating propulsion.

---

## Physics Model

The nanoparticle is modeled using overdamped Langevin dynamics.

### Translational dynamics

$$
d\mathbf{r}
=
v_p \hat{\mathbf{n}}(\theta)\,dt
+
\sqrt{2D_T}\,d\mathbf{W}_T
$$

### Rotational dynamics

$$
d\theta
=
\sqrt{2D_R}\,dW_R
$$

where:

- $\mathbf{r}=(x,y)$ is the particle position
- $\theta$ is the particle orientation
- $v_p$ is the propulsion velocity
- $D_T$ is the translational diffusion coefficient
- $D_R$ is the rotational diffusion coefficient
- $\mathbf{W}_T$ and $W_R$ are Wiener processes

The propulsion velocity is controlled by the laser.

When the laser is OFF:

$$
v_p = 0
$$

When the laser is ON:

$$
v_p = v_{max}
$$

---

## Physical Parameters

| Parameter | Symbol | Value | Description |
|---|---:|---:|---|
| Boltzmann constant | $k_B$ | $1.38\times10^{-23}$ J/K | Thermal energy scale |
| Temperature | $T$ | 298 K | Simulation temperature |
| Dynamic viscosity | $\eta$ | $10^{-3}$ Pa·s | Water viscosity |
| Particle radius | $R$ | 33 nm | Particle radius |
| Translational diffusion | $D_T$ | $6.49\times10^{-12}$ m²/s | Translational diffusion |
| Rotational relaxation time | $\tau_R$ | 0.220 ms | Rotational timescale |
| Rotational diffusion | $D_R$ | $1/\tau_R$ | Rotational diffusion |
| Target Péclet number | $Pe$ | 0.30 | Dimensionless propulsion parameter |
| Maximum velocity | $v_{max}$ | calculated | Maximum propulsion speed |

The maximum propulsion velocity is calculated from:

$$
v_{max}
=
Pe\sqrt{D_TD_R}
$$

---

## RL Environment Design

### State Space

The agent receives two state variables:

```text
state = [normalized_distance, normalized_angle]