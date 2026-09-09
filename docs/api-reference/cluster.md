# Clustering

Leak-safe clustering for panels, ported from the Sov.ai SDK. Two modes are
supported. `KShapeClusterer` clusters each entity's **time-series shape** using
the k-Shape algorithm (FFT normalized cross-correlation + Shape-Based Distance);
centroids are learned on the training rows and frozen, so per-`(entity, time)`
distance-to-centroid columns and the hard cluster label are `leakage_safe`
features that refit per fold. `CrossSectionalClusterer` clusters **entities by
their same-date feature vectors** (KMeans / agglomerative) with a scaler fit on
the training rows only.

The k-Shape engine is a dependency-free NumPy port (no torch); `sbd` and `ncc`
are exposed as public utilities, and `k_from_n_entities` is a simple heuristic
for choosing the cluster count.

## What's here

| Your problem | Entry point |
| --- | --- |
| Group entities by the *shape* of their series | `KShapeClusterer` |
| Group entities by their same-date feature vectors | `CrossSectionalClusterer` |
| Shape-Based Distance / normalized cross-correlation | `sbd`, `ncc` |
| How many clusters for this many entities? | `k_from_n_entities` |

## See also

- [`reduce`](reduce.md) — leak-safe dimensionality reduction and latent factors.
- [Leak-safety](../concepts/leak-safety.md) — why centroids must be frozen at `fit` time.

## API

::: panelary.cluster
