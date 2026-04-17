# CDSF-Net: Fine-Grained Dating of Ancient Murals

Official implementation for **“Fine-Grained Dating of Ancient Murals with Degradation Adaptation and Forgetting-Aware Long-Tail Learning”**. The project focuses on **automatic dynasty/period classification for ancient mural images** under three practical challenges: **subtle inter-period style differences**, **spatially non-uniform degradation**, and **long-tailed data distribution**.  

## Overview

Ancient mural dating is an important problem in digital cultural heritage analysis. Traditional dating methods rely heavily on expert knowledge and are often subjective, time-consuming, and labor-intensive. This work formulates mural dating as a **fine-grained image classification task** and proposes **CDSF-Net**, a unified framework designed for degraded and imbalanced mural datasets.  

CDSF-Net is built to address three domain-specific difficulties:

* **Fine-grained stylistic differences** between adjacent historical periods
* **Non-uniform mural degradation**, such as fading, blur, and peeling
* **Long-tailed class imbalance** caused by uneven mural survivorship across dynasties 

## Main Contributions

* Proposes **CDSF-Net**, a mural dating framework that jointly models fine-grained discrimination, degradation robustness, and long-tail learning. 
* Builds **Mural5376**, a dataset of **5,376 mural images** spanning **10 Chinese historical periods**.  
* Introduces:

  * **CSFR**: Cross-Scale Feature Refinement
  * **DASF**: Degradation-Adaptive Semantic Fusion
  * **FCB**: Forgetting-aware Class-Balancing 

## Method

The model uses a **ConvNeXt-Base** hierarchical backbone and consists of three key modules: 

### 1. CSFR: Cross-Scale Feature Refinement

CSFR refines higher-level features with low-level details so that subtle stylistic cues, such as contours, brushwork, and clothing patterns, are better preserved for dynasty discrimination. 

### 2. DASF: Degradation-Adaptive Semantic Fusion

DASF adaptively fuses **local texture information** and **global semantic/style information** according to preservation quality. In well-preserved regions, local details receive more attention; in severely degraded regions, the model relies more on global context. 

### 3. FCB: Forgetting-aware Class-Balancing

FCB dynamically reweights training samples based on class imbalance and forgetting events, helping the model focus on underrepresented classes and hard boundary samples. 

The framework diagram on **page 8** visually summarizes how these three modules correspond to the three domain challenges. 

## Dataset

**Mural5376** contains **5,376 images** from the following 10 periods:
Jin-16 Kingdoms, Northern Wei, Western Wei, Northern Qi, Northern Zhou, Sui, Tang, Five Dynasties, Song, and Yuan. 

The sample distribution is highly imbalanced. For example, the paper notes that the **Tang** class contains far more samples than minority classes such as **Northern Qi**, which motivates the long-tail learning strategy. 

The dataset construction pipeline includes:

* document layout parsing,
* image-text matching,
* dynasty label extraction,
* deduplication,
* and quality control. 

## Results

On Mural5376, CDSF-Net achieves:

* **Top-1 Accuracy:** 88.97%
* **Top-3 Accuracy:** 98.39%
* **Macro-F1:** 0.8899 

Compared with the ConvNeXt-Base baseline, CDSF-Net improves **Top-1 accuracy by 3.34 percentage points**. 

The paper also shows that CDSF-Net is more robust under simulated degradation, including:

* blur,
* fading,
* peeling,
* and composite degradation scenarios. 

## Highlights

* Strong performance on **fine-grained dynasty classification**
* Better robustness under **severe mural degradation**
* Improved recognition of **minority and hard classes**
* A practical framework for **digital cultural heritage analysis** beyond murals, with possible transfer to paintings, textiles, and related artifacts  

## Citation

If you use this work, please cite the paper:

```bibtex
@article{wang2026cdsfnet,
  title={Fine-Grained Dating of Ancient Murals with Degradation Adaptation and Forgetting-Aware Long-Tail Learning},
  author={Jin Wang and Zheng Liu and Juan Wang and Xuelong Li and Yu Weng}
}
```

