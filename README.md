# Physics-Informed DQN for Janus Nanoparticle Navigation

This project investigates the navigation of an active Janus nanoparticle using Deep Reinforcement Learning (DQN) in a stochastic environment.

The nanoparticle is modeled as an active Brownian particle undergoing translational and rotational Brownian motion, with its self-propulsion controlled by a laser.

One of the most important properties of active particles is that their diffusion coefficient is not described solely by the conventional Einstein relation.

## Project Overview

The reinforcement learning agent observes the particle's normalized distance and angular difference relative to a target, and chooses between two discrete actions:

* `0` — Laser OFF
* `1` — Laser ON

A Double DQN architecture with experience replay and a target network is used to learn a temporal control strategy for reaching the target.

## Architecture

<img src="Janus architecture.png" alt="Reinforcement Learning Architecture" width="600">

## Physical Model

The particle dynamics are described by overdamped Langevin equations.

**Translational Dynamics:** The position of the particle evolves according to:

$$
dx = v_p \cos(\theta) \, dt + \sqrt{2D_T} \, dW_T^x
$$

$$
dy = v_p \sin(\theta) \, dt + \sqrt{2D_T} \, dW_T^y
$$

**Rotational Dynamics:** The orientation angle evolves as:

$$
d\theta = \sqrt{2D_R} \, dW_R
$$

where \( dW_T^x \), \( dW_T^y \), and \( dW_R \) are independent Wiener processes representing thermal fluctuations. The self-propulsion speed \( v_p \) is controlled by the laser, which the reinforcement learning agent can switch on or off to navigate toward the target.

The effective diffusion coefficient of the active particle is given by:

$$
D_{\mathrm{active}} = D_{\mathrm{HBM}} + \frac{1}{4}v^2\tau_R
$$

where \( D_{\mathrm{HBM}} \) represents the passive Brownian diffusion, \( v \) is the self-propulsion velocity, and \( \tau_R \) is the rotational relaxation time.

The rotational diffusion coefficient is related to the rotational relaxation time by:

$$
D_R = \frac{1}{\tau_R}
$$

The simulation uses physical parameters corresponding to a nanoscale Janus particle, including translational diffusion, rotational diffusion, particle size, and a target Péclet number.

## Reinforcement Learning

The learned policy is evaluated against simple control strategies such as:

* Always ON
* Always OFF
* Random ON/OFF

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

This project was developed to study the application of reinforcement learning to stochastic active-matter systems.
