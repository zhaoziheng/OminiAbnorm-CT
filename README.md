# OminiAbnorm-CT

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)]([TODO](https://www.arxiv.org/abs/2506.03238))
[![HF](https://img.shields.io/badge/Hugging%20Face-Data-yellow)](https://huggingface.co/datasets/zzh99/OminiAbnorm-CT-14K)
[![HF](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](TODO)

This is the official repository for "Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach".
**OminiAbnorm-CT** is developed for grounded abnormality analysis on CT images from multiple planes and all human body regions.

![Example Figure](docs/resources/model.png)

It is built on **OminiAbnorm-CT-14K**, first large-scale dataset designed for abnormality grounding and description on multi-plane whole-body CT imaging. 
It contains 14.5K CT images with grounding annotation for 19K abnormal findings.
Each abnormal findings is further linked to the detailed description in the report, and categorized according to a comprehensive hierarchical taxonomy.

![Example Figure](docs/resources/data.png)

🔍 Check our [paper]([TODO](https://www.arxiv.org/abs/2506.03238)) for more details.

⌛️ The code and model checkpoints will be released in the next few days. Stay tunned!

## Data
Check [here](https://huggingface.co/datasets/zzh99/OminiAbnorm-CT-14K) for the OminiAbnorm-CT-14K dataset and the taxonomy.

## Requirements
TBD

## Inference Guidance
TBD

## Traning Guidance
TBD

## Citation
```
@misc{zhao2025rethinkingwholebodyctimage,
      title={Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach}, 
      author={Ziheng Zhao and Lisong Dai and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2025},
      eprint={2506.03238},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.03238}, 
}
```
