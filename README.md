# (Custom) OpenAI Baselines

This is a forked version of https://github.com/openai/baselines. The forking
is dated March 21st, 2019; with last commit set to `1b09243`.

## Overview

The purpose of this forked version is to implement [Learning
to reinforcement learn](https://arxiv.org/abs/1611.05763) by Wang et al., since no official implementation is provided by DeepMind. Our implementation is partially dependent on the [NASGym](https://github.com/gomerudo/nas-env).

In the paper, a *deep meta-reinforcement learning* algorithm is implemented using the A2C setting, therefore we perform some modification on top of the A2C implementation on this baselines to come up with the meta version.

We hope that this code can be generalized to different environments that
implement the OpenAI Gym Environment interface too.
