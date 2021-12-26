---
title: "Whole-system multidimensional financial time-series modeling from mostly-homogeneous data"
author: Davide Roznowicz, Emanuele Ballarin
geometry: "top=2cm,bottom=2cm"
---

<br>
<br>

### Introduction

Since their very inception, financial markets have always drawn the interest and efforts of *modeling professionals* due to the obvious economic appeal they exhibit. Soon, in addition to *material goals*, their more abstract and academic study began being driven by the still ongoing endavour to prove or disprove their learnability via *statistically-learning machines*.  

The many different scales (micro- and macro-economic, and everything in between), items (time series of prices, volumes, etc... of different commodities and financial instruments, or economic agents themselves) and approaches (plain prediction, causal modeling, credit-assignment based, agent-based) of such modeling - and the numerosity and generally strong financial involvement of modelers themselves - contributed to a great abundance of models, of which - though - it is often difficult to grasp clear applicability boundaries, limits, computational resources necessary and actual performance (let alone the availability of their parameters, or weights) - and a relative scarcity of published scholarly articles or papers detailing their inner workings.  

Of those many models proposed to date, the most common category attacks the problem of predicting the time-dependent evolution of prices (or functions thereof) of tradeable items (e.g. currencies, commodities, stock shares) phenomenologically, via both classical statistical tools tailored toward time series modeling and *deep learning based* approaches. In any case, being the system extremely complex, given the lack of widely established *physical laws* governing it and the etherogeneity of their agents' motives, its very nature calls for following the most *holistic* and *data-driven* modeling approach possible.

\pagebreak

### Abstract

In our work we assess the feasibility, and produce a *proof of concept* implementation, of a *deep-learning-based* *end-to-end* system to predict or simulate multidimensional mostly-homogeneous financial time-series data with associated uncertainty quantification. The main focus will be on US stock shares prices.  

The choice of multidimensionality arises from the fact that - from a domain-specific standpoint - the information contained in unidimensional time series is severely limited, and prevents full exploitation of cross-correlations, reciprocal influences and variables conveying systemic effects.  

Given the *borderline-chaoticity* and *noisiness* of the system, but also the possibility of long-ranging dependencies, such previously-mentioned correlations, and the eventuality of sharply localized *phase transitions* in its dynamics, no *single-architecture* system seems right-off adequate for the task of interest.  

Neither:

- <u>Recurrent sequence models</u> (*RNNs and variations*), the usual choice for the modeling of medium-length, 1D, time-dependent but generally noiseless sequences,

- <u>Hierarchical convolutional cascades</u>, a relatively new but well-performing approach to short-term pattern or highly structured time-dependent modeling,

- <u>Token transformers</u> (*and dense attentional architectures in general*) which, though appealing from the modeling standpoint, strongly suffer from input-wise quadratic complexity, and have been originally developed for sequences of *discrete, dictionarized tokens*.

*Hybrid* or *evolved architectures*, such as *(hierarchical) CNN-RNN*, *CNN-Transformers* with sparse attention, or even more *radical* proposals, match more closely the ideal requirements. They blend classical time-dependent analysis with *differentiable programming blocks* (e.g. *FT-Transformers*, *Wavelet-CNN*), but lack extensive testing in time series modeling - particularly in the financial domain.

Driven by genuine curiosity for *the new*, and in an attempt to slightly tighten the gap between promising novel proposals and real-world battle-testing, our work will be centered around this last class of models. Among those, priority will be given to architectures trying to decouple *local, short patterns* (effectively modeled via convolution) and *global, more far-reaching trends* (modeled with sparse self-attention). Additionally, *variationalization* of predictions may provide the uncertainty quantification sought.  

Given the relative novelty of such experimentation, expected results are hard to figure at this time, being open to the widest variability in outcomes. Our optimistic hope is that such work could help to shed some fleeble light on the nature of the learnability of a financial system and to suggest potential paths to tread (or not to!) for its modeling with cutting edge deep learning technologies.

<br>
<br>

#### Minimal Bibliography

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, et al., 2017)

- [Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/abs/1907.00235) (Li et al., 2020)

- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (Kitaev et al., 2020)

- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) (Lee-Thorp et al., 2021)
