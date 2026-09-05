<div align="center">

# CDSF-Net

Core implementation of **CDSF-Net** for fine-grained dating of ancient murals.

ConvNeXt-Base backbone with `CSFR`, `DASF`, and `FCB`, organized into a clean standalone package for public release.

</div>

## Overview

- `CSFR`: Cross-Scale Feature Refinement for preserving subtle stylistic cues across feature levels
- `DASF`: Degradation-Adaptive Semantic Fusion for balancing local texture and global semantics
- `FCB`: forgetting-aware class balancing for long-tailed training

## Architecture

<p align="center">
  <img src="fig/fig1_overall.png" alt="CDSF-Net overall architecture" width="100%">
</p>


## Citation

```bibtex
@article{wang2026cdsfnet,
  title={Fine-Grained Dating of Ancient Murals with Degradation Adaptation and Forgetting-Aware Long-Tail Learning},
  author={Jin Wang and Zheng Liu and Juan Wang and Xuelong Li and Yu Weng}
}
```
