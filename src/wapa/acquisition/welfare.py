"""Label-free parameter and objective definitions shared with evaluation."""

import math


def resolve_parameters(users, *, rho=1.0, epsilon=0.5, gamma=1.0, target_group=None):
    if not math.isfinite(rho) or rho > 1:
        raise ValueError("rho must be finite and <= 1")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(gamma) or gamma < 1:
        raise ValueError("gamma must be finite and >= 1")
    if target_group is not None and (
        not isinstance(target_group, (list, tuple))
        or not all(isinstance(user, str) for user in target_group)
    ):
        raise ValueError("target_group must be a list of persona IDs")
    group = list(target_group or [])
    if len(set(group)) != len(group) or set(group) - set(users):
        raise ValueError("target_group contains duplicate or unknown personas")
    if not group and gamma != 1:
        raise ValueError("gamma != 1 requires a nonempty target_group")
    return dict(rho=rho, epsilon=epsilon, gamma=gamma, target_group=sorted(group))


def social_weight(user, parameters):
    return parameters["gamma"] if user in parameters["target_group"] else 1.0
