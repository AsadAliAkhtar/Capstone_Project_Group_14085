"""
Root-level re-export so notebooks can do `from msvar import em_fit`
after adding the project root to sys.path.
"""
from src.msvar import (
    build_regressors,
    emission_logprob,
    hamilton_filter,
    kim_smoother,
    mstep_var_params,
    mstep_transition,
    ergodic_distribution,
    em_fit,
    MSVARResult,
    simulate_msvar,
)
