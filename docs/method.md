# Acquisition and welfare

For a frozen base-model distribution `p` and the reference-ensemble mean `q`, WAPA estimates normalized Brier loss as

```text
s = (1 + sum(p²) - 2 sum(p q)) / 2.
```

For a binary comparison with original-A probabilities `p` and `q`, this is `q(1-p)² + (1-q)p²`.

For M comparisons per persona, estimated supervision utility is `U_i = 1 - sum(unqueried scores)/M`. The denominator stays M throughout selection. A queried comparison contributes zero supervision loss because its label is observed.

Welfare is a weighted sum of `h_rho(U_i + epsilon)`, where `h_rho(x) = x^rho/rho` for nonzero rho and `log(x)` at zero. Persona weights are gamma within the declared Target group and 1 elsewhere. Normalizing weights to sum to one does not change the selected set.

WAPA selects by the exact finite increment:

```text
weight_i × [h_rho(U_i + epsilon + s/M) - h_rho(U_i + epsilon)].
```

Within each persona, scores are sorted descending. A heap merges the decreasing marginal sequences, updating only the selected persona. Under frozen predictions, equal comparison counts, nonnegative gains and rho <= 1, this maximizes the estimated objective at the fixed budget. The implementation evaluates increments with `log1p` and `expm1` to avoid cancellation.

## Baselines

- **Random** uses seeded sampling without replacement, independent of welfare settings.
- **Entropy** uses `H(p)/log(K)`.
- **Reference Disagreement** uses mean squared distance from reference vectors to their mean, divided by `1 - 1/K`. For two options this equals variance of reference A probabilities divided by `0.25`.

The weighted baselines use their own normalized scores both for remaining persona risk and cell gain. They retain the first-order allocation rule `weight_i × (U_i + epsilon)^(rho-1) × score/M`. At rho=gamma=1, weighted methods reduce to global score ranking.

## Reference conditions and evaluation

No queries retains base-model probabilities everywhere. All pseudo-labels trains on those probabilities; All ground-truth labels trains on revealed labels everywhere.

The Welfare-optimal Oracle runs exact allocation with realized normalized Brier losses as gains. It uses the full label set to optimize the configured scalar welfare objective and is implemented in the evaluation module.

Observed persona utility is one minus mean normalized Brier loss. Queried cells contribute zero loss. Mean averages personas equally; Tail averages the bottom 10%, using a fractional boundary persona when needed; Target averages a fixed supplied cohort. Policy evaluation computes these metrics from the trained model's DPO-implied preference probabilities on test comparisons.
