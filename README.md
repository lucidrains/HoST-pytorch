<img src="./rewards.png" width="600px"></img>

## HoST - Pytorch (wip)

Implementation of Humanoid Standing Up, from the paper <a href="https://taohuang13.github.io/humanoid-standingup.github.io/">Learning Humanoid Standing-up Control across Diverse Postures</a> out of Shanghai, in Pytorch

Besides for the set of reward functions, the other contribution is validating [an approach using multiple critics](https://openreview.net/forum?id=rJvY_5OzoI) out of Boston University

## Install

```bash
$ pip install HoST-pytorch
```

## Usage

```python
import torch
from host_pytorch import Agent
from host_pytorch.mock_env import Env, mock_hparams

env = Env()

agent = Agent(
    actor = dict(
        dims = (env.dim_state, 256, 128),
        num_actions = 5,
    ),
    critics = dict(
        dims = (env.dim_state, 256),
        num_actions = 5
    ),
    reward_hparams = mock_hparams()
)

memories = agent(env)

agent.learn(memories)

agent.save('./standing-up-policy.pt', overwrite = True)
```

## Citations

```bibtex
@article{huang2025host,
  title     = {Learning Humanoid Standing-up Control across Diverse Postures},
  author    = {Huang, Tao and Ren, Junli and Wang, Huayi and Wang, Zirui and Ben, Qingwei and Wen, Muning and Chen, Xiao and Li, Jianan and Pang, Jiangmiao},
  journal   = {arXiv preprint arXiv:2502.08378},
  year      = {2025},
}
```

```bibtex
@article{Farebrother2024StopRT,
    title   = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
    author  = {Jesse Farebrother and Jordi Orbay and Quan Ho Vuong and Adrien Ali Taiga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
    journal = {ArXiv},
    year   = {2024},
    volume = {abs/2403.03950},
    url    = {https://api.semanticscholar.org/CorpusID:268253088}
}
```

```bibtex
@article{Tao2022LearningTG,
    title  = {Learning to Get Up},
    author = {Tianxin Tao and Matthew Wilson and Ruiyu Gou and Michiel van de Panne},
    journal = {ACM SIGGRAPH 2022 Conference Proceedings},
    year   = {2022},
    url    = {https://api.semanticscholar.org/CorpusID:248496244}
}
```
