"""
This code is based on the example from:
Cuong Duong. "Introduction to Bayesian A/B Testing". In: PyMC Examples.
Ed. by PyMC Team. DOI: 10.5281/zenodo.5654871.
Modified by Fabio Brady, 2024.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='pytensor.tensor.blas')
RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)


@dataclass
class BetaPrior:
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def plot(self, x_range: np.array = None, threshold: float = 1e-6, **kwargs):
        """Plot the Beta distribution for conversion rate prior."""
        if x_range is None:
            x_range = np.linspace(0, 1, 100_000)

        y = beta.pdf(x_range, self.alpha, self.beta)

        mask = y > threshold
        x_filtered = x_range[mask]
        y_filtered = y[mask]

        plt.plot(x_filtered, y_filtered, label=f"Beta({self.alpha}, {self.beta})", **kwargs)
        plt.title("Conversion Rate Prior")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


@dataclass
class BinomialData:
    trials: int
    successes: int


@dataclass
class GammaPrior:
    def __init__(self, alpha: float, beta: float, name: str):
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def plot(self, x_range: np.array = None, threshold: float = 1e-6, **kwargs):
        """Plot the Gamma distribution for shape/rate prior, filtering out low PDF values."""
        if x_range is None:
            x_range = np.linspace(0, 10, 1_000_000)  # Use the instance's default x_values

        y = gamma.pdf(x_range, self.alpha, scale=1 / self.beta)

        # Filter the x_range and y values where the PDF is above the threshold
        mask = y > threshold
        x_filtered = x_range[mask]
        y_filtered = y[mask]

        plt.plot(x_filtered, y_filtered, label=f"Gamma({self.alpha}, {self.beta})", **kwargs)
        plt.title(f"{self.name} Prior")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


@dataclass
class ConversionData:
    visitors: int
    conversions: int
    name: str


@dataclass
class RevenueData:
    visitors: int
    purchased: int
    total_revenue: float
    name: str


class ConversionRateModel:
    def __init__(self, conversion_rate_prior: BetaPrior):
        self.conversion_rate_prior = conversion_rate_prior
        self.models = []
        self.comparison_method = ""
        # self.theta = np.array([])
        # self.theta_uplift = np.array([])
        # self.reluplift = np.array([])

    def create_model(self, data: List[ConversionData], comparison_method: str) -> "ConversionRateModel":
        self.comparison_method = comparison_method

        # Create the models based on the data
        for variant in data:
            alpha_posterior = self.conversion_rate_prior.alpha + variant.conversions
            beta_posterior = self.conversion_rate_prior.beta + variant.visitors - variant.conversions
            self.models.append(beta(alpha_posterior, beta_posterior))

        return self

    def sample(self, samples: int) -> Dict[str, List[np.ndarray]]:
        # Reset theta, theta_uplift, and reluplift to avoid accumulation across runs
        self.theta = []
        self.theta_uplift = []
        self.reluplift = []

        # Sample from each beta distribution and store in self.theta
        self.theta = [model.rvs(samples) for model in self.models]

        if self.comparison_method == 'compare_to_control':
            control_theta = self.theta[0]  # Control
            for i in range(1, len(self.models)):
                uplift = self.theta[i] - control_theta
                relative_uplift = uplift / control_theta
                self.theta_uplift.append(uplift)
                self.reluplift.append(relative_uplift)

        elif self.comparison_method == 'best_of_rest':
            # Find the best-performing variant (including control)
            max_theta = np.max(self.theta, axis=0)  # Best model at each sample, including control

            for i in range(len(self.models)):  # Compare each variant, including control, to the best of the rest
                uplift = self.theta[i] - max_theta
                relative_uplift = uplift / max_theta
                self.theta_uplift.append(uplift)
                self.reluplift.append(relative_uplift)

        else:
            raise ValueError('Not recognized comparison method')

        return {'theta': self.theta, 'theta_uplift': self.theta_uplift, 'theta_reluplift': self.reluplift}