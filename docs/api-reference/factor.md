# Cross-sectional / factor evaluation

Leak-safe tools for turning a signal into an evaluated cross-sectional factor.
`forward_return` is the single audited place where a forward return is
constructed — a backward per-entity shift with a gap guard — so alignment leaks
cannot creep in elsewhere. `ic` / `ic_summary` compute the per-date information
coefficient (rank or Pearson) and summarise it (mean IC, ICIR, t-stat,
hit-rate). `portfolio_sort` forms per-date quantile buckets and reports the
long-short spread and monotonicity. `orthogonalize` de-correlates a feature set
per date (Gram-Schmidt / QR), distinct from the Numerai-style residual
neutralization in `.xs.neutralize`.

All operations are strictly per-date (never global) and lean on the mandatory
`over=` cross-sectional key for their leak-safety.

::: panelary.factor
