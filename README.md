# HRes-Adapter

<p align="center">
  <img src="https://img.shields.io/badge/Model-UniXcoder--base-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Trainable%20Params-0.958%25-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Vulnerability%20Detection-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

---

## Overview

**HRes-Adapter** (Hierarchical Residual Stream Adapter) is a novel Parameter-Efficient Fine-Tuning (PEFT) framework built on top of [UniXcoder-base](https://huggingface.co/microsoft/unixcoder-base) for source code vulnerability detection. It trains only **0.958%** of the total model parameters while achieving balanced, generalizable predictions across multiple vulnerability benchmarks.

> **Trained model weights are available on HuggingFace:**
> ### 🔗 [[https://huggingface.co/An26745asdg/Unixcoder_HRes_PEFT](https://huggingface.co/An26745asdg/Unixcoder_HRes_PEFT/upload/main](https://huggingface.co/An26745asdg/Unixcoder_HRes_PEFT/tree/main))

---

## Methodology




## Algorithm

```
Input:  X ∈ R^{B×T×d}, frozen encoder M with L layers
Trainable: {Wd, Wu, Φ_res, B_res, α_res, Wc}

1.  M ← (1 − m) × −10000 ;  Q←XWQ,  K←XWK,  V←XWV
2.  S ← softmax(QKᵀ/√dh + M) ;  A ← SV
3.  ΔA ← (S̄ · XWd)Wu ;  A ← WO(A + ΔA)
4.  X_mid ← LayerNorm(X + A)
5.  H_raw ← α_res(X̂Φ_res) + B_res ;  H̄ ← mean_{B,T}(H_raw)
6.  H_res ← SinkhornKnopp(exp(H̄), iters=20)
7.  X_enh ← einsum(H_res, split(X_mid, 2))
8.  F ← W_FFN^out · GELU(W_FFN^in · X_mid)
9.  X_out ← LayerNorm(X_mid + X_enh + F) ;  repeat ∀ l ∈ L
10. Ŷ ← X_out[:,0,:] · Wc

Output: Ŷ ∈ R^{B×2}  (vulnerability prediction logits)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/HRes-Adapter-PEFT.git
cd HRes-Adapter-PEFT


**Requirements:**
```
torch>=2.0.0
transformers>=4.35.0
pandas
scikit-learn
numpy
tqdm
matplotlib
```



