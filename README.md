# Code

This repository contains the code that is able to reproduce the results and figures for: 

> Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers, 
> by Freya Behrens, Luca Biggio, Lenka Zdeborová

## Reproducing Figures

The results are decentrally saved in a wandb repository that is accessible to the public:
```
- feeds/phase_diagram_T32
- feeds/phase_diagram_T32_L15
- feeds/random_phase_diagram_T32
- feeds/phase_diagram_T64
```
The learning process is documented with the weights along the training.
The notebooks to create the figures rely on downloading these results.
The notebooks are named according to the figure number in the paper, which should help identifying the results you are interested in.

## Reproducing the Results

The results were generated from scripts, that can be generated from ```[experiments]-scripts.ipynb```.
These should be configured with your personal wandb id. 
Running all experiments takes approximately 1 week on a single GPU (NVIDIA RTX A5000).

## Explicit Constructions

In ```[theory]-explicit-constructions-d=T.ipynb``` and its dependencies we implemented the algorithms from out explicit constructions for $d=T$.
In ```[theory]-explicit-constructions-d<T-mutual-coherence.ipynb``` and its dependencies we implemented the algorithms from out explicit constructions for $d<T$ that rely on ideas connected to the mutual coherence bounds.
In ```[theory]-explicit-constructions-d<T-softmax.ipynb``` and its dependencies we implemented the algorithms from out explicit constructions for $d<T$ that rely on ideas connected to the softmax.

## Software requirements

```
wandb
torch
numpy
matplotlib
seaborn
tqdm
```

## Questions and Contact

If you have any questions, feel free to contact us through the email adresses linked on the paper, or simply create an issue on this repo.