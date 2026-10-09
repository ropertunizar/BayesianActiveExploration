"""
Area Under the Sparsification Error (AUSE, Gustafsson et al. 2020): how well an uncertainty score ranks the
prediction errors. Samples are removed in decreasing order of uncertainty (model curve) or of error (oracle
curve); the AUSE is the area between both curves. Lower is better.
"""
import numpy as np


def compute_ause(error, uncertainty):
    """
    Args:
        error: (n,) prediction error of each sample
        uncertainty: (n,) uncertainty score of each sample

    Returns:
        oracle sparsification curve, model sparsification curve, AUSE
    """
    error = np.asarray(error, dtype=np.float64)
    rescaled_error = error / np.sum(error)
    n = len(error)

    error_sorted = rescaled_error[np.argsort(rescaled_error)[::-1]]
    error_sorted_per_uncertainty = rescaled_error[np.argsort(np.asarray(uncertainty))[::-1]]

    # remaining error after removing the k most erroneous / most uncertain samples, k = 0 ... n - 1
    scurve_oracle = np.cumsum(error_sorted[::-1])[::-1]
    scurve_model = np.cumsum(error_sorted_per_uncertainty[::-1])[::-1]

    ause = np.abs(np.trapz(scurve_oracle, dx=1 / n) - np.trapz(scurve_model, dx=1 / n))
    return scurve_oracle, scurve_model, ause


def plot_ause(scurve_oracle, scurve_model, ause, title, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, len(scurve_oracle))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'{title}  AUSE: {ause:.3f}', fontsize=14)
    ax.plot(x, scurve_oracle, color='black', label='oracle')
    ax.plot(x, scurve_model, color='tab:blue', linestyle='dashed', label='model')
    ax.fill_between(x, scurve_oracle, scurve_model, alpha=0.1, color='tab:gray', label='AUSE')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Fraction of samples removed', fontsize=12)
    ax.set_ylabel('MAE (rescaled)', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
