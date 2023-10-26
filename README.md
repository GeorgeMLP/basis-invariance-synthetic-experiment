# A Synthetic Experiment for Basis invariance

Basis invariance synthetic experiment in Appendix D of the paper "**[Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding](https://openreview.net/forum?id=1mAYtdoYw6)**", NeurIPS 2023.

This experiment tests the performance of GNN models under *basis ambiguity*. We use graph isomorphic testing, a traditional graph task. Our focus is on 10 non-isomorphic random weighted graphs $`\mathcal{G}_1,\dots,\mathcal{G}_{10}`$, all exhibiting basis ambiguity issues (with the first three eigenvectors belonging to the same eigenspace). We sample 20 instances for each graph, introducing different permutations and basis choices for the initial eigenspace. The dataset is then split into a 9:1 ratio for training and testing, respectively. The task is a 10-way classification, where the aim is to determine the isomorphism of a given graph to one of the 10 original graphs. The model is given the first 3 eigenvectors as input.

Test accuracy reported in our paper:

| Positional Encoding | Accuracy        |
| ------------------- | --------------- |
| LapPE               | 0.11 ± 0.08     |
| LapPE + random sign | 0.10 ± 0.09     |
| LapPE + SignNet     | 0.10 ± 0.03     |
| LapPE + MAP (ours)  | **0.84 ± 0.21** |
