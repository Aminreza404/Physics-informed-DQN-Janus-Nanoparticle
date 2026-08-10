# Physics-Informed DQN for Janus Nanoparticle Navigation

This project investigates the navigation of an active Janus nanoparticle using Deep Reinforcement Learning (DQN) in a stochastic environment.

The nanoparticle is modeled as an active Brownian particle undergoing translational and rotational Brownian motion, with its self-propulsion controlled by a laser.

A key feature of active particles is that their **effective diffusion** can contain contributions beyond the conventional thermal Brownian diffusion. In this model, the effective translational diffusion is written as:

$$
D_{\mathrm{active}} =
D_{\mathrm{HBM}} +
\frac{1}{4}v^2\tau_R
$$

where (D_{\mathrm{HBM}}) represents the passive Brownian diffusion, (v) is the self-propulsion velocity, and (\tau_R) is the rotational relaxation time.

The rotational diffusion coefficient is related to the rotational relaxation time by:

$$
D_R = \frac{1}{\tau_R}
$$

## Project Overview

The reinforcement learning agent observes the particle's normalized distance and angular difference relative to a target, and chooses between two actions:

* `0` — Laser OFF
* `1` — Laser ON

A Double DQN architecture with experience replay and a target network is used to learn a temporal control strategy for reaching the target.

<p align="center">
  <img src="Janus Architecture.png" width="700">
</p>

## Physical Model

The particle dynamics are described by overdamped Langevin equations, incorporating both deterministic self-propulsion and stochastic Brownian motion.

The translational dynamics are given by:

$$
d\mathbf{r}
===========

v\hat{\mathbf{n}}(\theta),dt
+
\sqrt{2D_T},d\mathbf{W}_T
$$

and the rotational dynamics by:

$$
d\theta
=======

\sqrt{2D_R},dW_R
$$

The simulation uses physical parameters corresponding to a nanoscale Janus particle, including translational diffusion, rotational diffusion, particle size, propulsion velocity, and a target Péclet number.

## Reinforcement Learning

The learned policy is evaluated against simple control strategies such as:

* Always ON
* Always OFF
* Random ON/OFF

This allows the learned DQN policy to be compared with simple baseline strategies for controlling the active nanoparticle.

## Technologies

* Python
* NumPy
* PyTorch
* Matplotlib
* Computational Physics
* Deep Reinforcement Learning

## Physical Model Reference

The physical model and physical parameters used for the Janus nanoparticle are based on:

**"Light-Activated Self-Thermophoretic Janus Nanopropellers"**

[arXiv:2602.17548](https://arxiv.org/abs/2602.17548)

The Langevin dynamics, particle properties, diffusion parameters, and propulsion-related physical assumptions were adapted from the physical model presented in this work.

The reinforcement learning environment, DQN implementation, reward function, training procedure, and simulation code were implemented for this project.

## Main File

`Janus.py`

This project was developed to study the application of deep reinforcement learning to stochastic active-matter systems and the control of active Janus nanoparticles.

