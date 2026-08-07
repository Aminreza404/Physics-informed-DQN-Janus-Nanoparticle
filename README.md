# Physics-Informed DQN for Janus Nanoparticle Navigation

This project investigates the navigation of an active Janus nanoparticle using Deep Reinforcement Learning (DQN) in a stochastic environment.

The nanoparticle is modeled as an active Brownian particle undergoing translational and rotational Brownian motion, with its self-propulsion controlled by a laser.

The particle dynamics are described by overdamped Langevin equations:

$$
d\mathbf{r}
===========

v_p\hat{\mathbf{n}}(\theta),dt
+
\sqrt{2D_T},d\mathbf{W}_T
$$

$$
d\theta
=======

\sqrt{2D_R},dW_R
$$

The reinforcement learning agent observes the particle's normalized distance and angular difference relative to a target, and chooses between two actions:

* `0` — Laser OFF
* `1` — Laser ON

A Double DQN architecture with experience replay and a target network is used to learn a temporal control strategy for reaching the target.

The simulation uses physical parameters corresponding to a nanoscale Janus particle, including translational diffusion, rotational diffusion, particle size, and a target Péclet number.

The learned policy is evaluated against simple control strategies such as always ON, always OFF, and random ON/OFF.

## Technologies

* Python
* NumPy
* PyTorch
* Matplotlib
* Computational Physics
* Deep Reinforcement Learning

## Main File

`ENCR_DQN.ipynb`

This project was developed for studying the application of reinforcement learning to stochastic active-matter systems.

