# (Custom) OpenAI Baselines

This is a forked version of https://github.com/openai/baselines. The forking
is dated March 21st, 2019; with last commit set to `1b09243`.

## Purpose

The purpose of this forked version is - mainly - to implement the "Learning
to reinforcement learn" paper from Wang et al., since no official implementation
is provided by DeepMind.

In their paper (see https://arxiv.org/abs/1611.05763), the authors use a Deep
Meta-Reinforcement Learning algorithm based on Actor-Critic algorithms: A2C/A3C.
Therefore, we take this code as starting point to modify the official A2C
implementation provided by OpenAI, so that we can end into a Deep Meta-Reinforcement
Learning version of A2C, by preserving the main aspects of the base architcture.

We hope that this code can be generalized to different environments that
implement the OpenAI Gym Environment interface.