# Mathematical specification codebook

## Purpose

The record documents what is present in a method's published mathematical specification. It does not assign a method to one exclusive family and does not infer a biological claim from an algorithm name.

## Coding unit

The coding unit is one published dimensionality-reduction method. Each row must be supported by its primary publication and the corresponding derivation in Supplementary Note 1.

## Fields

| Field | Coding rule |
| --- | --- |
| `method` | Published method name used in the manuscript. |
| `navigation_group` | Historical reading-order label only. It is not used as an inferential class. |
| `observation_model` | Distribution or loss applied to the observed expression values. Use `not_explicit` when no observation model is specified. |
| `latent_parameterization` | How latent coordinates are represented, such as an orthogonal factor, probabilistic factor, Gaussian process, neural encoder, spectral coordinates or directly optimized coordinates. |
| `explicit_objective_terms` | Semicolon-separated controlled terms that appear explicitly in the objective or generative factorisation. |
| `constructed_or_conditioning_structures` | Inputs or intermediate structures used to define the objective, such as a count matrix, zero mask, k-nearest-neighbour graph, labels, batch identifiers, kernels, triplets or diffusion operator. |
| `directly_constrained_object` | The mathematical object whose discrepancy, likelihood or ordering is directly optimized. |
| `not_established_by_objective` | Properties that cannot be concluded from the objective alone and therefore require empirical validation. |
| `derivation_anchor` | Heading in Supplementary Note 1 containing the formula used for coding. |
| `verification_status` | Provenance state of the row. |

## Controlled objective terms

| Term | Inclusion criterion |
| --- | --- |
| `reconstruction` | The objective explicitly compares observed and reconstructed expression or features. |
| `likelihood` | The method optimizes a likelihood, deviance, evidence lower bound or explicit probabilistic observation model. |
| `latent_prior` | A prior or divergence term explicitly regularizes the latent distribution. |
| `zero_model_or_imputation` | Zero inflation, a zero gate, dropout model or iterative imputation is explicit in the model or objective. |
| `graph_regularization` | Adjacency, graph Laplacian or graph reconstruction appears explicitly in the objective. |
| `pairwise_probability` | High- and low-dimensional pair probabilities or memberships are matched directly. |
| `pairwise_attraction_repulsion` | Sampled pair classes are optimized through explicit attraction or repulsion terms without a probability-distribution or rank-order objective. |
| `diffusion_distance` | A diffusion operator or diffusion-derived distance defines the fitted geometry. |
| `metric_stress` | Distances are matched through an MDS-like stress or squared discrepancy. |
| `relative_order` | Triplet, quadruplet or rank/order constraints are optimized. |
| `similarity_learning` | A cell-cell similarity matrix is estimated as an optimization variable. |
| `clustering_regularization` | Mixture, entropy or clustering structure appears explicitly in the training objective. |
| `batch_alignment` | Batch labels or cross-batch relations explicitly enter the objective or sampled constraints. |
| `distribution_matching` | An adversarial, MMD or related discrepancy aligns distributions. |
| `sparsity_regularization` | A sparsity penalty explicitly contributes to the objective. |

## Adjudication rules

1. Code only terms that can be located in an equation, generative factorisation or algorithmic step in the primary paper.
2. Record a graph under `constructed_or_conditioning_structures` when it defines pairs or targets but is not itself reconstructed or penalized.
3. Do not translate a property into a biological claim. For example, a neighbourhood loss constrains selected neighbour relations; it does not establish cell identity.
4. When a software implementation differs from the publication, code the publication here and record implementation deviations in a separate reproducibility table.
5. Multi-stage methods receive all terms that influence the final representation. The sequence of stages must be stated in `directly_constrained_object`.
6. For final human verification, two coders should independently assess all categorical fields. Agreement should be calculated before adjudication, with disagreements and the final rationale retained in the audit record.
