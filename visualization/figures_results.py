import numpy as np

from matplotlib import pyplot as plt


# Figure. Here we study how the composition of censored and
# uncensored patients during training impacts the C-index mean ±
# standard error over the 5 fold in the SUPPORT2 dataset. The
# validation and test sets are fixed and the training set has censored
# patients introduced by marking patients as censored at random.
# The plot starts at 30% because the dataset has that many censored
# patients by default. We find that the WM classification loss is
# robust to the introduction of censored data.
x = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

WM_mean = np.array([85.33, 83.00, 83.04, 82.74, 82.60, 82.88, 82.78])
WM_std = np.array([0.52, 1.19, 1.28, 1.28, 1.67, 1.21, 1.36]) / np.sqrt(5)

sigm_mean = np.array([85.53, 82.65, 82.42, 82.33, 82.37, 82.29, 81.94])
sigm_std = np.array([1.25, 1.44, 1.33, 1.35, 1.22, 1.53, 1.02]) / np.sqrt(5)

cox_efron_mean = np.array([84.91, 83.22, 83.17, 83.18, 82.81, 82.80, 82.88])
cox_efron_std = np.array([1.34, 1.25, 1.17, 1.51, 1.43, 1.31, 1.29]) / np.sqrt(5)

plt.rcParams['figure.figsize'] = (5, 3)
plt.errorbar(x, cox_efron_mean, yerr=cox_efron_std, label="Cox Efron", color="#1B2ACC", linestyle="--")
plt.errorbar(x, sigm_mean, yerr=sigm_std, label="Ranking", color="#CC4F1B", linestyle=":")
plt.errorbar(x, WM_mean, yerr=WM_std, label="WM (ours)", color="#3F7F4C", linestyle="-.")

plt.xlabel("% censured patients")
plt.ylabel("C-index")
plt.legend()
plt.tight_layout()
plt.show()
