# Meta-A2C

This is a change on top of the A2C algorithm from the OpenAI baselines as of commit `1b09243`. For now, the code is customized to work with the [NASGym](https://github.com/gomerudo/nas-env).

- Original meta-A2C paper: https://arxiv.org/abs/1611.05763
- Original A2C paper: https://arxiv.org/abs/1602.01783
- Example of usage: `python -m baselines.run --alg=meta_a2c --env=PongNoFrameskip-v4 --network=meta_lstm [...]`.

## Changes in files with respect to the original A2C
- `policies.py`: Accepts more inputs for the *meta* approach
- `meta_a2c.py`: Instantiates the meta-network and runs adapts the batches of experiments to provide the *meta* inputs.
- `runner.py`: Generates the *meta* inputs.
