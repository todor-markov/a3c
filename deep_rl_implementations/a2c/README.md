# A2C
An implementation of the Advantage Actor-Critic (A2C) algorithm - a synchronous adaptation of the Asynchronous Advantage Actor Critic (A3C) algorithm originally described in [Mnih et al. 2015][1]

This implementation was inspired by the OpenAI Baselines [implementation][2] of the same algorithm, and I have borrowed multiple pieces from there (mostly around gym environment wrappers)

[1]: https://arxiv.org/pdf/1602.01783.pdf
[2]: https://github.com/openai/baselines/tree/master/baselines/a2c