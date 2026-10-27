# GLM-3: Geometric Language Model

### Exploring Three-Dimensional Geometric Representations for Language Modeling

[![Research](https://img.shields.io/badge/Research-Experimental-000000)](https://github.com/Ahmadjamil888/GLM-3)
[![Architecture](https://img.shields.io/badge/Architecture-Geometric-333333)](https://github.com/Ahmadjamil888/GLM-3)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-555555)](https://github.com/Ahmadjamil888/GLM-3)

**GLM-3 (Geometric Language Model)** is a proposed language-model architecture exploring whether geometric relationships between contextual token representations can provide a useful alternative to conventional dot-product attention mechanisms.

The central hypothesis is that language-model interactions can be formulated using learned geometric relationships in a three-dimensional Euclidean representation space. Rather than treating token representations solely as high-dimensional feature vectors, GLM-3 investigates the role of spatial relationships, transformed distances, and geometric compatibility in contextual information processing.

This repository documents the architecture, its mathematical formulation, implementation approach, research motivation, experimental requirements, and known limitations.

**Research principle:** Define the architecture precisely, evaluate it rigorously, and let experimental evidence determine its value.

---

## Table of Contents

- [Research Paper](#research-paper)
- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Research Objectives](#research-objectives)
- [Core Architectural Idea](#core-architectural-idea)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Mathematical Formulation](#mathematical-formulation)
- [Geometric Relation Mechanism](#geometric-relation-mechanism)
- [GLM-3 Processing Pipeline](#glm-3-processing-pipeline)
- [Worked Numerical Example](#worked-numerical-example)
- [Causal Information Flow](#causal-information-flow)
- [Three-Dimensional Representation](#three-dimensional-representation)
- [Output Decoder](#output-decoder)
- [Training Objective](#training-objective)
- [Reference Implementation](#reference-implementation)
- [Installation and Environment](#installation-and-environment)
- [Prototype Usage](#prototype-usage)
- [Experimental Research Plan](#experimental-research-plan)
- [Baselines and Comparisons](#baselines-and-comparisons)
- [Ablation Studies](#ablation-studies)
- [Evaluation Metrics](#evaluation-metrics)
- [Computational Complexity](#computational-complexity)
- [Potential Applications](#potential-applications)
- [Limitations and Open Problems](#limitations-and-open-problems)
- [Related Research](#related-research)
- [Research Roadmap](#research-roadmap)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Research Paper

The full research paper describes the proposed architecture, mathematical formulation, geometric interaction mechanism, worked examples, prototype implementation, experimental methodology, limitations, and related research.

### Read the paper online

**[Open the complete GLM-3 research paper (PDF)](https://github.com/Ahmadjamil888/GLM-3/blob/main/download.pdf)**

#### Embedded PDF viewer

The following viewer uses Google's document viewer with the repository's raw PDF file as its source.

<iframe src="https://raw.githubusercontent.com/Ahmadjamil888/GLM-3/refs/heads/main/GLM-3_%20Geometry%20in%20Language.png" width="100%" height="720" style="border: 1px solid #d1d5db;" title="GLM-3 Research Paper PDF Viewer"></iframe>

If the embedded viewer is unavailable in your Markdown renderer, use one of these alternatives:

- **GitHub PDF page:** [View download.pdf](https://github.com/Ahmadjamil888/GLM-3/blob/main/download.pdf)
- **Direct PDF download:** [Download the research paper](https://github.com/Ahmadjamil888/GLM-3/raw/refs/heads/main/download.pdf)
- **Raw PDF source:** [Open the PDF file](https://raw.githubusercontent.com/Ahmadjamil888/GLM-3/main/download.pdf)

**Embedding note:** GitHub's README renderer may remove iframe elements for security reasons. The iframe is intended for compatible Markdown renderers. If you want a guaranteed embedded viewer on a public website, deploy a dedicated GitHub Pages page using Mozilla PDF.js and point it to `download.pdf`.

---

## Overview

Modern language models use learned representations to encode information about tokens, context, syntax, semantics, and relationships. Transformer architectures have made attention-based contextual processing a foundational technique in contemporary language modeling.

GLM-3 explores a different design constraint: **what if every contextual token state were represented strictly in three-dimensional Euclidean space, and token interactions were computed through learned geometric relationships?**

The architecture proposes a sequence of operations:

1. Map input tokens into three-dimensional vectors.
2. Add learned positional representations.
3. Compute pairwise geometric compatibility scores.
4. Apply causal masking to prevent access to future tokens.
5. Normalize compatibility scores into attention-like weights.
6. Aggregate information from earlier token states.
7. Update contextual states through residual connections and nonlinear transformations.
8. Convert the resulting three-dimensional states into vocabulary logits.
9. Train the model using next-token prediction.

GLM-3 retains the general pattern of pairwise interaction and weighted information aggregation, while investigating a distance-based compatibility function and a strict three-dimensional contextual-state constraint.

This makes GLM-3 a testable architectural hypothesis rather than an established replacement for Transformer models.

### Project objectives

| Objective | Description |
|---|---|
| Geometric representation | Constrain contextual token states to three dimensions. |
| Relational processing | Compute interactions through transformed geometric distances. |
| Autoregressive prediction | Preserve causal information flow for next-token generation. |
| Mathematical clarity | Define each transformation and its dimensions explicitly. |
| Experimental evaluation | Compare the architecture against controlled baselines. |
| Reproducibility | Document the training process, measurements, and limitations. |

---

## Research Motivation

A representation space is more than a place to store numbers. Its geometry can influence which relationships are easy for a neural model to express and learn.

In conventional attention, compatibility is commonly computed using learned query and key projections, followed by a scaled dot product. GLM-3 investigates whether a transformed Euclidean distance can serve as an alternative compatibility measure.

The motivation is to examine three related questions.

### 1. Can geometric compatibility guide contextual information flow?

If two states have a small transformed distance, a distance-based scoring function can assign them a larger compatibility score. Training may learn arrangements in which useful contextual dependencies become geometrically accessible.

However, proximity does not inherently mean semantic relevance. The model must learn useful representations through training.

### 2. Can strict three-dimensional states represent useful linguistic information?

A three-dimensional state is compact and easy to visualize, but it has limited representational capacity compared with wider hidden states.

GLM-3 treats this limitation as part of the research question. The aim is not to assume that three dimensions are sufficient, but to determine which tasks they can support and where they fail.

### 3. Does geometric scoring provide a measurable advantage?

A meaningful evaluation must compare geometric scoring with other mechanisms under controlled conditions. A visually interpretable representation is not sufficient evidence of better language modeling.

The architecture must be evaluated using held-out tasks, language-modeling metrics, compute measurements, and multiple random seeds.

---

## Research Objectives

The current research direction can be organized into three levels of evidence.

### Objective A: Establish trainability

Determine whether the proposed model can learn simple causal dependencies, predict held-out patterns, and maintain stable optimization.

### Objective B: Evaluate the geometric inductive bias

Determine whether distance-based scoring performs better than alternative scoring mechanisms when the contextual-state width and training conditions are held constant.

### Objective C: Evaluate practical competitiveness

Determine whether the strict three-dimensional model can approach conventional language-model baselines on meaningful tasks and whether its compact state representation yields practical efficiency benefits.

These objectives are distinct. Success at the first level does not establish success at the second or third.

---

## Core Architectural Idea

GLM-3 represents token states as points in a three-dimensional vector space.

For a sequence of tokens,

\[
x_1,x_2,\ldots,x_L,
\]

the model maintains a contextual state for each position:

\[
h_i \in \mathbb{R}^{3}.
\]

The three-dimensional restriction applies to every contextual state throughout the model's geometric blocks. It is not merely a projection used to visualize a higher-dimensional internal representation.

A geometric relation score is calculated between a query state and a transformed source state:

\[
s_{ij}
=
-\frac{\left\|h_i-Rh_j\right\|_2^2}{\tau}.
\]

Here:

- \(h_i\) is the state at the query position.
- \(h_j\) is the state at a candidate source position.
- \(R\) is a learned linear transformation.
- \(\tau\) is a positive temperature parameter.
- \(\|\cdot\|_2^2\) denotes squared Euclidean distance.

A smaller transformed distance produces a larger score. The scores are normalized across permitted causal positions to obtain weights.

The resulting weights determine how much information is aggregated from each source state.

### Important terminology

The scoring function is a **transformed-distance compatibility function**. When \(R\) is learned and applied asymmetrically, the expression is not necessarily a metric over the original token states: symmetry and the triangle inequality are not guaranteed.

This distinction matters when interpreting the geometry mathematically.

---

## Architecture at a Glance

The baseline architecture follows this processing pipeline:

```text
Input Token Sequence
        |
        v
Token Embedding Table
        |
        v
3D Token Representations
        |
        +---- Positional Embeddings
        |
        v
Initial Contextual States in R^3
        |
        v
Geometric Relation Scoring
        |
        v
Causal Masking
        |
        v
Softmax Normalization
        |
        v
Weighted Message Aggregation
        |
        v
Nonlinear State Update
        |
        v
Residual Connection and Normalization
        |
        +---- Repeated Geometric Blocks
        |
        v
3D Contextual States
        |
        v
Nonlinear Vocabulary Decoder
        |
        v
Next-Token Probability Distribution
```

The main architectural constraint is that the contextual state entering and leaving every geometric block remains three-dimensional.

The decoder can use a wider intermediate layer to map the state into vocabulary logits. That does not remove the information bottleneck at the contextual state itself.

---

## Mathematical Formulation

### 1. Token embedding

Let the vocabulary contain \(V\) tokens. The token embedding matrix is:

\[
E\in\mathbb{R}^{V\times3}.
\]

For token \(x_i\), its embedding is:

\[
e_i=E[x_i]\in\mathbb{R}^{3}.
\]

Each token therefore begins with three learned embedding coordinates.

### 2. Positional embedding

Token order matters in language. GLM-3 uses a learned positional embedding table:

\[
P\in\mathbb{R}^{L_{\max}\times3}.
\]

The initial state at position \(i\) is:

\[
h_i^{(0)}=E[x_i]+P[i].
\]

Both terms have dimensionality three, so their sum remains in \(\mathbb{R}^{3}\).

### 3. Geometric relation transform

For each candidate source state, apply a learned transformation:

\[
k_j=Rh_j,
\qquad
R\in\mathbb{R}^{3\times3}.
\]

The transformed difference is:

\[
d_{ij}=h_i-Rh_j.
\]

Its squared distance is:

\[
D_{ij}=\|d_{ij}\|_2^2.
\]

The compatibility score becomes:

\[
s_{ij}=-D_{ij}/\tau.
\]

The negative sign makes smaller distances correspond to larger scores.

### 4. Causal attention weights

For an autoregressive model, position \(i\) may use only positions \(j\le i\).

The normalized weights are:

\[
\alpha_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{k\le i}\exp(s_{ik})},
\qquad j\le i.
\]

For future positions \(j>i\), the score is masked to negative infinity before softmax, producing zero attention weight.

The weights for each valid query position sum to one.

### 5. Message aggregation

Let \(A\) be a learned message transformation:

\[
A\in\mathbb{R}^{3\times3}.
\]

Transform each source state:

\[
v_j=Ah_j.
\]

Aggregate the source messages:

\[
m_i=\sum_{j\le i}\alpha_{ij}Ah_j.
\]

Because each transformed message is three-dimensional and the weights are scalar values, the aggregate remains in \(\mathbb{R}^{3}\).

### 6. Nonlinear update

A small nonlinear function \(U\) updates the state:

\[
U:\mathbb{R}^{3}\rightarrow\mathbb{R}^{3}.
\]

A residual update with normalization can be expressed as:

\[
h_i^{(\ell+1)}
=
\operatorname{LN}
\left(
h_i^{(\ell)}+U(m_i)
\right).
\]

The block preserves three-dimensional contextual states across layers.

The exact placement of normalization, residual paths, and nonlinearities should be tested through ablation studies rather than assumed to be optimal.

---

## Geometric Relation Mechanism

The main difference between GLM-3 and conventional dot-product attention lies in the compatibility function.

A conventional scaled dot-product mechanism has the general form:

\[
s_{ij}^{\mathrm{dot}}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}.
\]

The GLM-3 baseline instead uses:

\[
s_{ij}^{\mathrm{geo}}
=
-\frac{\|h_i-Rh_j\|_2^2}{\tau}.
\]

The two scoring functions have different inductive biases.

| Property | Dot-product scoring | GLM-3 distance scoring |
|---|---|---|
| Compatibility basis | Vector alignment | Transformed squared distance |
| Learned projections | Query and key projections | Geometric relation transform |
| Score interpretation | Similarity or alignment | Negative transformed distance |
| Causal masking | Required for autoregressive use | Required for autoregressive use |
| Weighted aggregation | Commonly used | Used |
| Pairwise score matrix | Yes | Yes |
| Contextual-state width in baseline | Usually configurable and wider | Strictly three |

This comparison does not imply that one method is inherently better. Their relative performance is an empirical question.

---

## Worked Numerical Example

Consider a query state:

\[
q=(1,0,0)
\]

and two candidate source states:

\[
a=(0,0,0),
\qquad
b=(1,0,0).
\]

For this illustration, let \(R=I\) and \(\tau=1\).

The squared distance from \(q\) to \(a\) is:

\[
D(q,a)=(1-0)^2=1.
\]

The squared distance from \(q\) to \(b\) is:

\[
D(q,b)=(1-1)^2=0.
\]

The scores are therefore:

\[
s_a=-1,
\qquad
s_b=0.
\]

After softmax normalization:

\[
\alpha_a
=
\frac{e^{-1}}{e^{-1}+e^0}
\approx0.269,
\]

\[
\alpha_b
=
\frac{e^0}{e^{-1}+e^0}
\approx0.731.
\]

The second candidate receives the larger weight because its transformed distance is smaller.

This example demonstrates the scoring calculation only. It does not establish semantic understanding: if the closer state is irrelevant to the prediction, the resulting weights may be unhelpful.

---

## GLM-3 Processing Pipeline

### Step 1: Tokenization

Convert raw text into token identifiers using a tokenizer.

The tokenizer defines the vocabulary and determines how text is segmented. The vocabulary size must match the model's embedding table and decoder output.

### Step 2: Embedding initialization

Map each token identifier and position to three-dimensional vectors.

### Step 3: Pairwise geometric scoring

Compute transformed distances between the query states and permitted source states.

### Step 4: Causal masking

Prevent each position from accessing future positions.

### Step 5: Normalized aggregation

Apply softmax to the allowed scores and aggregate transformed source states.

### Step 6: State update

Apply a nonlinear transformation, residual connection, and normalization.

### Step 7: Vocabulary prediction

Map each final contextual state to vocabulary logits and normalize them into a probability distribution.

### Step 8: Optimization

Compare predicted next-token probabilities with the actual target tokens and update the model parameters through backpropagation.

---

## Causal Information Flow

For autoregressive language modeling, the prediction at position \(i\) must depend only on the current token and its permitted history.

The model should satisfy:

\[
p(x_{i+1}\mid x_1,\ldots,x_i).
\]

It must not access \(x_{i+1}\) or later tokens while computing the prediction at position \(i\).

Causal masking must be applied before softmax. Otherwise, future tokens could influence earlier predictions during training, producing invalid evaluation results.

A key implementation test is to modify future tokens while keeping the prefix fixed. The logits at earlier positions must remain unchanged in evaluation mode.

---

## Three-Dimensional Representation

The three-dimensional constraint is central to the GLM-3 research hypothesis.

For every position \(i\):

\[
h_i=(h_{i,1},h_{i,2},h_{i,3}).
\]

These coordinates are learned latent features. They do not inherently correspond to predefined concepts such as subject, verb, tense, sentiment, or entity identity.

The model must discover useful arrangements through its training objective.

### Potential benefits

- Compact contextual states.
- A simple and interpretable geometric scoring equation.
- A useful experimental setting for studying representational bottlenecks.
- Lower per-pair feature computation than a wider representation, all else equal.
- Opportunities to visualize learned states and distance distributions.

### Potential disadvantages

- Limited representational capacity.
- Possible overlap between contexts that require different predictions.
- Sensitivity to learned scales and distance distributions.
- Potential state collapse or oversmoothing across layers.
- Difficulty representing multiple independent linguistic factors simultaneously.

A low-dimensional representation may be useful for certain controlled tasks without being sufficient for general-purpose language modeling.

---

## Output Decoder

After the geometric blocks, each contextual state remains in \(\mathbb{R}^{3}\).

A nonlinear decoder can map this state into vocabulary logits:

\[
z_i
=
W_2\operatorname{GELU}(W_1h_i+b_1)+b_2.
\]

Here, \(W_1\) can map from three dimensions to a wider intermediate representation, while \(W_2\) maps the intermediate representation to \(V\) vocabulary logits.

The next-token distribution is:

\[
p(x_{i+1}\mid x_{\le i})
=
\operatorname{softmax}(z_i).
\]

A wider decoder increases the complexity of the output mapping, but it cannot recover information lost when different contexts are compressed into the same final three-dimensional state.

This distinction is important when evaluating whether the model's limitations arise from the geometric score function or from the strict state-width constraint.

---

## Training Objective

GLM-3 can be trained using the standard autoregressive next-token prediction objective.

Given a sequence:

\[
x_1,x_2,\ldots,x_T,
\]

the model receives the input tokens up to position \(T-1\) and predicts the targets from positions \(2\) through \(T\).

The negative log-likelihood objective is:

\[
\mathcal{L}(\theta)
=
-\sum_{t=1}^{T-1}
\log p_\theta(x_{t+1}\mid x_{\le t}).
\]

In practice, the loss is typically averaged over valid target tokens. Padding positions should be excluded from the loss.

### Optimization considerations

A controlled training setup should document:

- Optimizer and learning rate.
- Batch size and sequence length.
- Training-token count and number of updates.
- Learning-rate schedule and warm-up.
- Random seeds and parameter initialization.
- Gradient clipping, if used.
- Validation and checkpoint-selection rules.
- Hardware, runtime, and memory consumption.

The temperature should remain positive. One common parameterization is:

\[
\tau=\operatorname{softplus}(\rho)+\epsilon.
\]

The model should also be monitored for unstable scores, vanishing gradients, exploding gradients, and state collapse.

---

## Reference Implementation

The following PyTorch implementation illustrates the core geometric block and a minimal autoregressive model. It is a research prototype, not a complete production training system.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricBlock(nn.Module):
    def __init__(self, dim=3):
        super().__init__()

        self.relation = nn.Linear(
            dim, dim, bias=False
        )
        self.message = nn.Linear(
            dim, dim, bias=False
        )

        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

        self.norm = nn.LayerNorm(dim)
        self.rho = nn.Parameter(
            torch.tensor(0.54)
        )

    def forward(self, x):
        # x: [batch, sequence, 3]
        _, length, _ = x.shape

        source = self.relation(x)

        delta = (
            x.unsqueeze(2)
            - source.unsqueeze(1)
        )

        scores = -delta.square().sum(dim=-1)

        temperature = (
            F.softplus(self.rho) + 1e-4
        )
        scores = scores / temperature

        future_mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            future_mask,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        values = self.message(x)
        message = weights @ values

        return self.norm(
            x + self.update(message)
        )


class GeometricLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length=128,
        layers=4,
        decoder_width=64,
    ):
        super().__init__()

        self.token = nn.Embedding(
            vocab_size, 3
        )
        self.position = nn.Embedding(
            max_length, 3
        )

        self.blocks = nn.ModuleList([
            GeometricBlock(3)
            for _ in range(layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(3, decoder_width),
            nn.GELU(),
            nn.Linear(decoder_width, vocab_size),
        )

    def forward(self, token_ids):
        batch, length = token_ids.shape

        if length > self.position.num_embeddings:
            raise ValueError(
                "Sequence exceeds maximum length."
            )

        positions = torch.arange(
            length,
            device=token_ids.device,
        )

        x = (
            self.token(token_ids)
            + self.position(positions)[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        return self.decoder(x)


def next_token_loss(model, tokens):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits = model(inputs)

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
```

### Implementation notes

This implementation is intentionally compact. Before using it for meaningful benchmarks, address the following:

1. Add padding-aware attention masks and ignore padded targets in the loss.
2. Test causal masking automatically.
3. Verify output shapes, finite losses, and gradients.
4. Record the exact parameter count and configuration.
5. Add a tokenizer and a documented data pipeline.
6. Implement validation, checkpoint saving, and reproducible evaluation.
7. Benchmark memory and runtime on the target hardware.

The code illustrates one baseline formulation. It does not establish that the architecture is optimal, novel, or competitive.

---

## Installation and Environment

The reference implementation requires Python and PyTorch.

### 1. Clone the repository

```bash
git clone https://github.com/Ahmadjamil888/GLM-3.git
cd GLM-3
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install PyTorch

Install a PyTorch build appropriate for your operating system and hardware by following the official instructions:

https://pytorch.org/get-started/locally/

For a CPU-only environment, a basic installation may be sufficient:

```bash
pip install torch
```

The precise dependencies and commands should be updated to match the actual files and tested configuration in this repository.

---

## Prototype Usage

A minimal smoke test can verify that the model accepts token identifiers and returns vocabulary logits.

```python
import torch

vocab_size = 100
model = GeometricLM(
    vocab_size=vocab_size,
    max_length=128,
    layers=4,
)

tokens = torch.randint(
    0, vocab_size, (2, 17)
)

loss = next_token_loss(model, tokens)
loss.backward()

print("Loss:", loss.item())
```

For the configuration shown, the loss function uses 16 input positions to predict 16 next-token targets per sequence.

The expected logits shape for an input tensor of shape `[batch, length]` is:

```text
[batch, length, vocab_size]
```

This is a smoke test, not a benchmark. Random tokens do not demonstrate language understanding or successful training.

---

## Experimental Research Plan

A rigorous evaluation should test the architecture in stages.

### Stage 1: Synthetic dependency tasks

Use controlled tasks where the correct relationship is known.

Examples include:

- Copy and repeat patterns.
- Controlled token retrieval.
- Simple bracket matching.
- Subject–verb agreement.
- Dependency retention across distractor tokens.
- Compositional combinations held out during training.

Measure performance as sequence length and dependency distance increase.

### Stage 2: Small-scale language modeling

Evaluate on a documented text corpus with a fixed tokenizer and train/validation/test split.

Report validation cross-entropy and perplexity, along with the number of training tokens and optimization steps.

### Stage 3: Generalization tests

Evaluate whether the model generalizes to:

- Unseen token combinations.
- Longer contexts than those encountered during training.
- Distractors inserted between related tokens.
- Changes in word order.
- Controlled examples where similar surface patterns require different predictions.

### Stage 4: Efficiency measurements

Measure wall-clock time, throughput, peak memory, and model parameters under clearly specified hardware and training conditions.

Low dimensionality alone does not prove lower end-to-end latency or superior efficiency.

---

## Baselines and Comparisons

GLM-3 should be compared against baselines that isolate the source of any observed performance difference.

| Baseline | Purpose |
|---|---|
| GLM-3 with geometric scoring | Tests the primary formulation. |
| Three-dimensional dot-product attention | Tests whether geometric distance helps under the same state-width constraint. |
| Three-dimensional MLP pair scorer | Tests whether the exact distance score is important. |
| Three-dimensional recurrent model | Tests whether pairwise aggregation is necessary. |
| Wider conventional Transformer | Measures the cost of the strict three-dimensional bottleneck. |
| Wider geometric models | Tests how results change with representation width. |

Comparisons should use consistent tokenization, data splits, training tokens, optimization budgets, and evaluation procedures wherever possible.

Parameter-matched and compute-matched experiments should be reported separately because they answer different questions.

---

## Ablation Studies

Ablations help identify which architectural decisions contribute to the results.

| Ablation | Research question |
|---|---|
| Identity versus learned relation transform | Does learning the transform help? |
| Fixed versus learned temperature | Does adaptive score sharpness improve optimization? |
| Distance versus dot-product scoring | Does geometric compatibility add value? |
| Position embeddings enabled versus disabled | How important is explicit position information? |
| Residual connection enabled versus disabled | Does the residual path help preserve state information? |
| Alternative normalization strategies | Which configuration is most stable? |
| Different numbers of blocks | Does depth improve performance or cause oversmoothing? |
| Linear versus nonlinear decoder | How much does decoder capacity affect results? |
| State widths of 3, 8, 16, and 32 | How strongly does the dimensionality bottleneck affect performance? |

The wider variants are diagnostic comparisons, not strict GLM-3 models under the original three-dimensional constraint.

---

## Evaluation Metrics

A complete evaluation should include predictive quality, efficiency, and geometric diagnostics.

### Predictive quality

- Next-token cross-entropy.
- Perplexity on held-out text.
- Accuracy on synthetic tasks.
- Exact-match performance where appropriate.
- Generalization performance on held-out combinations.

### Efficiency

- Parameter count.
- Training time and total compute.
- Tokens processed per second.
- Inference latency.
- Peak memory consumption.
- Estimated FLOPs under a documented counting convention.

### Geometric diagnostics

- State norm distributions.
- Pairwise-distance distributions.
- Relation-score distributions.
- Attention entropy.
- Effective number of attended positions.
- State similarity and collapse indicators.
- Sensitivity to perturbations in token context.

Use multiple random seeds and report variation rather than selecting only the best run.

---

## Computational Complexity

Let:

- \(B\) be the batch size.
- \(L\) be the sequence length.
- \(d\) be the contextual-state width, fixed at 3 in the baseline.
- \(N\) be the number of geometric blocks.
- \(V\) be the vocabulary size.

The pairwise geometric score computation requires approximately:

\[
O(BL^2d)
\]

operations per block.

Message aggregation also scales approximately as:

\[
O(BL^2d).
\]

The full score or weight matrix requires:

\[
O(BL^2)
\]

memory when materialized.

The three-dimensional representation reduces the per-pair feature width, but **does not remove the quadratic dependence on sequence length**.

The vocabulary decoder can also be computationally significant, especially when \(V\) is large.

Consequently, any efficiency claim must be supported by actual measurements against relevant baselines.

---

## Potential Applications

GLM-3 is currently best understood as an experimental architecture for studying geometric representation and low-dimensional contextual processing.

Potential research applications include:

- Investigating the relationship between geometric embeddings and contextual prediction.
- Studying the expressivity of severely constrained hidden states.
- Testing alternative compatibility functions for sequence modeling.
- Developing controlled experiments on relational and compositional generalization.
- Exploring interpretable diagnostics for learned vector spaces.
- Comparing geometric and conventional attention under matched constraints.

These are research directions, not claims that GLM-3 has already demonstrated practical advantages in these domains.

---

## Limitations and Open Problems

### 1. The three-dimensional information bottleneck

Each contextual state contains only three scalar values. Different contexts that map to the same final state cannot be distinguished by the decoder.

A nonlinear decoder can create complex decision boundaries, but it cannot recover contextual information that is absent from its input.

### 2. Geometric proximity does not guarantee relevance

The model learns to use distances as compatibility scores, but Euclidean proximity is not inherently equivalent to semantic similarity or causal relevance.

### 3. Quadratic sequence complexity

The pairwise interaction matrix still grows quadratically with sequence length. Three-dimensional states alone do not solve the memory and compute challenges of long-context processing.

### 4. Optimization stability

Temperature, initialization, normalization, and repeated aggregation may affect score distributions and state diversity. The model must be monitored for unstable gradients and representation collapse.

### 5. Architectural novelty remains to be established

Geometric embeddings, relational transformations, message passing, and attention mechanisms have substantial prior literature.

A new combination or dimensionality constraint may motivate useful experiments, but novelty requires a systematic literature review and a clear distinction from existing methods.

### 6. Performance remains unverified

The architecture specification and prototype do not, by themselves, demonstrate superiority over Transformer models.

Any claim of improved quality, efficiency, scalability, or generalization should be supported by reproducible measurements.

---

## Related Research

The following works provide useful foundations for understanding attention, geometric representations, and relational learning.

1. **Vaswani et al. (2017), Attention Is All You Need.** Introduces the Transformer architecture and scaled dot-product attention.  
   https://arxiv.org/abs/1706.03762

2. **Nickel and Kiela (2017), Poincaré Embeddings for Learning Hierarchical Representations.** Studies hyperbolic embeddings for hierarchical data.  
   https://arxiv.org/abs/1705.08039

3. **Sun et al. (2019), RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.** Investigates relation modeling through rotations in complex space.  
   https://arxiv.org/abs/1902.10197

4. **Thomas et al. (2018), Tensor Field Networks.** Studies rotation- and translation-equivariant neural networks for three-dimensional point clouds.  
   https://arxiv.org/abs/1802.08219

5. **Google Machine Learning Crash Course: Embeddings.** Provides background on vector representations and embedding spaces.  
   https://developers.google.com/machine-learning/crash-course/embeddings

These works motivate relevant comparisons but do not establish that GLM-3 is novel or that a strict three-dimensional representation is sufficient for general-purpose language modeling.

---

## Research Roadmap

The next steps should prioritize falsifiable experiments over architectural complexity.

### Phase 1: Baseline validation

- Implement the strict three-dimensional architecture.
- Add unit tests for tensor shapes and causal masking.
- Verify finite loss and gradients.
- Establish a reproducible training configuration.

### Phase 2: Controlled synthetic evaluation

- Train on simple dependency tasks.
- Test generalization across longer distances.
- Measure sensitivity to distractor tokens.
- Compare against a three-dimensional dot-product baseline.

### Phase 3: Representation analysis

- Inspect state norms and pairwise distances.
- Measure score entropy and state diversity.
- Identify possible collapse and aliasing.
- Compare linear and nonlinear decoders.

### Phase 4: Language-modeling evaluation

- Select a documented corpus and tokenizer.
- Establish train, validation, and test splits.
- Report cross-entropy and perplexity.
- Run multiple seeds and controlled baselines.

### Phase 5: Efficiency and publication

- Measure compute, memory, and latency.
- Publish configurations and evaluation scripts.
- Document negative results as well as positive findings.
- Complete a systematic novelty review before making novelty claims.

---

## Reproducibility

A result should be considered reproducible only when the experimental conditions are sufficiently documented.

The repository should aim to include:

- Model source code and configuration files.
- Environment and dependency instructions.
- Tokenizer configuration.
- Dataset sources and preprocessing procedures.
- Train/validation/test split definitions.
- Random seeds and optimizer settings.
- Training and evaluation scripts.
- Parameter counts and compute measurements.
- Unit tests for causal behavior.
- Checkpoints or instructions for regenerating them, where practical.
- A clear record of completed and planned experiments.

Do not report unmeasured performance values as experimental results.

---

## Contributing

Contributions are welcome in areas that improve the mathematical clarity, implementation correctness, or empirical evaluation of GLM-3.

Potential contributions include:

- Implementation and numerical-stability improvements.
- Causal-mask and gradient tests.
- Alternative geometric scoring functions.
- Controlled baseline implementations.
- Reproducible synthetic tasks.
- Benchmarking and efficiency analysis.
- Literature review and comparisons with related architectures.
- Documentation and visualization of learned representations.

For substantial changes, include the motivation, relevant equations, experimental setup, baseline comparisons, and limitations.

The objective is to improve the quality of the research, not simply to increase model complexity.

---

## Citation

If you reference this project, cite the repository and the accompanying research paper.

A provisional BibTeX entry is provided below. Replace the author, publication, and year fields as appropriate when a formal publication or versioned release is available.

```bibtex
@misc{glm3_geometric_language_model,
  title        = {GLM-3: Geometric Language Model},
  author       = {Ahmad Jamil},
  year         = {2026},
  howpublished = {\url{https://github.com/Ahmadjamil888/GLM-3}},
  note         = {Research prototype and technical documentation}
}
```

## License

The applicable license should be specified in the repository's `LICENSE` file. Until a license is present, users should not assume that the source code or research materials are available for unrestricted reuse, redistribution, or commercial use.

## Acknowledgements

GLM-3 builds upon the broader research tradition of neural language modeling, representation learning, geometric embeddings, and attention-based sequence processing.

The project is motivated by a continuing research question: whether changing the geometry and dimensionality of contextual representations can produce useful, measurable differences in how language models process information.

---

## Final Research Statement

GLM-3 proposes a concrete and testable architecture: contextual token states constrained to three-dimensional Euclidean space, transformed-distance compatibility scores, causal aggregation, nonlinear state updates, and next-token prediction.

Its mathematical feasibility makes it possible to implement and evaluate. Its practical value, representational adequacy, computational advantages, and novelty remain open questions.

The appropriate next step is rigorous experimentation: establish trainability, isolate the contribution of geometric scoring, measure the effect of the three-dimensional bottleneck, and compare results against controlled baselines.

**GLM-3 is an investigation into geometric language-model design, with its conclusions to be determined by evidence.**
