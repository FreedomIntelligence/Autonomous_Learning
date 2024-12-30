# Autonomous Learning (AL)
This is the official repository for the paper 'LLMs Could Autonomously Learn Without External Supervision'.
<div align=center>
<img src="./image/AL_Overview.png" width = "75%" alt="Autonomous Learning" align=center/>
</div>

## Requirements

* Python == 3.10.13
* torch == 2.0.1
* transformers == 4.35.0
* alignment-handbook == 0.3.0.dev0
* numpy == 1.26.3
* vllm == 0.2.1
* trl == 0.7.4
* accelerate == 0.23.2

## run
```
bash run_AL.sh
```


## Reproducibility
We experiment on 8 Tesla A100-80GB GPU.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Ke Ji (keji@link.cuhk.edu.cn).

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{ji2024llms,
  title={LLMs Could Autonomously Learn Without External Supervision},
  author={Ji, Ke and Chen, Junying and Gao, Anningzhe and Xie, Wenya and Wan, Xiang and Wang, Benyou},
  journal={arXiv preprint arXiv:2406.00606},
  year={2024}
}
```
