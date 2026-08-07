# Physics-Informed DQN for Janus Nanoparticle Navigation

This project investigates the navigation of an active Janus nanoparticle using Deep Reinforcement Learning (DQN) in a stochastic environment.

The nanoparticle is modeled as an active Brownian particle undergoing translational and rotational Brownian motion, with its self-propulsion controlled by a laser.

## Project Overview

<p align="center">
  <img src="Janus architecture.png" width="700">
</p>

The reinforcement learning agent observes the particle's normalized distance and angular difference relative to a target, and chooses between two actions:

* `0` — Laser OFF
* `1` — Laser ON

A Double DQN architecture with experience replay and a target network is used to learn a temporal control strategy for reaching the target.

## Physical Model

The particle dynamics are described by overdamped Langevin equations:

<p align="center">
  <b>Translational dynamics</b>
</p>

<p align="center">
  <i>d</i>r = <i>v</i><sub>p</sub> n&#770;(<i>&theta;</i>) <i>dt</i> + &radic;(2<i>D</i><sub>T</sub>) <i>dW</i><sub>T</sub>
</p>

<p align="center">
  <b>Rotational dynamics</b>
</p>

<p align="center">
  <i>d&theta;</i> = &radic;(2<i>D</i><sub>R</sub>) <i>dW</i><sub>R</sub>
</p>

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

