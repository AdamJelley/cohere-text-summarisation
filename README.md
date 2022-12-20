# Minimal Cohere Texrt Summarisation Application with FastAPI
Minimal [FastAPI](https://fastapi.tiangolo.com) webapp to serve text summaries from [Cohere](https://cohere.ai) text summarisation [endpoint](https://docs.cohere.ai/docs/text-summarization-guide).

## Installation
Only `cohere` and `fastapi[all]` packages are required. These can be installed with:

```
pip install -r requirements.txt
```

## Usage
Launch application from root with:
```
uvicorn main:app
```
with optional flag `--reload` for ongoing development.

## Example
With prompt taken as abstract from [Trasformers learn in-context by gradient descent](https://arxiv.org/abs/2212.07677v1):

> Transformers have become the state-of-the-art neural network architecture across numerous domains of machine learning. This is partly due to their celebrated ability to transfer and to learn in-context based on few examples. Nevertheless, the mechanisms by which Transformers become in-context learners are not well understood and remain mostly an intuition. Here, we argue that training Transformers on auto-regressive tasks can be closely related to well-known gradient-based meta-learning formulations. We start by providing a simple weight construction that shows the equivalence of data transformations induced by 1) a single linear self-attention layer and by 2) gradient-descent (GD) on a regression loss. Motivated by that construction, we show empirically that when training self-attention-only Transformers on simple regression tasks either the models learned by GD and Transformers show great similarity or, remarkably, the weights found by optimization match the construction. Thus we show how trained Transformers implement gradient descent in their forward pass. This allows us, at least in the domain of regression problems, to mechanistically understand the inner workings of optimized Transformers that learn in-context. Furthermore, we identify how Transformers surpass plain gradient descent by an iterative curvature correction and learn linear models on deep data representations to solve non-linear regression tasks. Finally, we discuss intriguing parallels to a mechanism identified to be crucial for in-context learning termed induction-head (Olsson et al., 2022) and show how it could be understood as a specific case of in-context learning by gradient descent learning within Transformers.

A summary generated by cohere text summarisation:

> TLDR: Transformers are the state-of-the-art for machine learning. This is because they can generalize to in-context situations.

## To Do
- Try generating multiple outputs and ranking by descending likelihood to improve quality.
- Allow variable number of tokens to enable adjustable summarisation length.
- Try adjusting other parameters such as temperature (degree of randomness), top-k and top-p output sampling, frequency and presence penalties to reduce repitition etc.
- Implement front end.
- Enable secret api keys.