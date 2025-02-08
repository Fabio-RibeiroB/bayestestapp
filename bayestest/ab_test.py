"""
Classes to set up A/B analysis and add variants
"""
from typing import Optional, Union
import pandas as pd
from .utils import *
from .bayesian_models import *


class BayesTest:
    def __init__(self, comparison_method: str = 'compare_to_control') -> None:
        self.data: List[Union[ConversionData, RevenueData]] = []  # This will store either type but not both
        self.conversion_rate_prior = None
        self.shape_prior = None
        self.rate_prior = None
        self.comparison_method = comparison_method
        self.results = None
        self.model = None
        if self.comparison_method not in ('compare_to_control', 'best_of_rest'):
            raise ValueError("Allowed comparison methods are compare_to_control and best_of_rest")

    def set_conversion_rate_prior(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Return a BetaPrior for conversion rate modeling."""
        self.conversion_rate_prior = BetaPrior(alpha=alpha, beta=beta)

    def set_shape_prior(self, alpha: float = 3.0, beta: float = 1.0) -> None:
        """Return a GammaPrior for shape parameter."""
        self.shape_prior = GammaPrior(alpha=alpha, beta=beta, name='Shape')

    def set_rate_prior(self, alpha: float = 10.0, beta: float = 10.0 / 10_000) -> None:
        """Return a GammaPrior for shape parameter."""
        self.rate_prior = GammaPrior(alpha=alpha, beta=beta, name='Rate')

    def add_variant(self, visitors: int, conversions: int, name: Optional[str] = None,
                    total_revenue: Optional[float] = None, control=False) -> None:
        """
        Add a variant to the test. Depending on whether total_revenue is provided, it adds either ConversionData or RevenueData.
        If there's a mix of data types, it raises an error.
        """
        # If total_revenue is None, we are adding ConversionData
        if total_revenue is None:
            if self.data and isinstance(self.data[0], RevenueData):
                raise TypeError("Cannot add ConversionData to a test that already contains RevenueData.")

            self.data.append(ConversionData(visitors=visitors, conversions=conversions, name=name))

        # If total_revenue is provided, we are adding RevenueData
        elif isinstance(total_revenue, (float, int)):
            if self.data and isinstance(self.data[0], ConversionData):
                raise TypeError("Cannot add RevenueData to a test that already contains ConversionData.")

            self.data.append(
                RevenueData(visitors=visitors, purchased=conversions, total_revenue=total_revenue, name=name))

        else:
            raise ValueError("Invalid total_revenue value. Must be a float or int.")

        if control:  # if control, move to start of data list, otherwise first added variant is always control
            self.data = [self.data[-1]] + self.data[:-1]

    def run(self, samples: int=50_000, yoy_visitors: Optional[int] = None, margin_rate: Optional[float] = 0.0537, chains: Optional[int] = 2) -> None:
        """
        Execute the Bayesian A/B test by creating the model and performing posterior sampling.

        This method sets up the Bayesian model using the provided data, then runs sampling to estimate
        the posterior distributions. It also calculates the expected uplift based on the results.

        :param samples: The number of samples to draw during the MCMC sampling process. Defaults to 100,000.
        :type samples: int, optional
        :param yoy_visitors: Year-over-year visitors or traffic data to estimate overall visitor growth 
            or to adjust the model. Defaults to None.
        :type yoy_visitors: int, optional
        :param margin_rate: The assumed margin rate for revenue calculations, representing the profit margin as 
            a decimal. Defaults to 0.0537 (5.37%).
        :type margin_rate: float, optional
        :param chains: The number of Markov Chains to run during sampling. More chains help with convergence diagnostics.
        :type chains: int, optional
        
        :return: None. The method updates the internal model and stores the resulting posterior distributions 
            and calculated uplift metrics.
        :rtype: None
        """

        if not self.data:
            raise ValueError("No data provided for inference. Please add variants before calling run.")

        if hasattr(self.data[0], 'total_revenue') and self.data[0].total_revenue is not None and yoy_visitors is None:
            raise ValueError("Missing argument: yoy_visitors")

        if isinstance(self.data[0], ConversionData):
            self.model = ConversionRateModel(self.conversion_rate_prior).create_model(self.data, self.comparison_method)
            results = self.model.sample(samples)
            self.results = results
            return

    def reset(self) -> None:
        """Resets variant data and models but not priors."""
        self.data = []
        self.model = None

    def posterior_samples(self) -> pd.DataFrame:
        """
        Returns a DataFrame where each column represents the samples for a parameter 
        (like theta, lam, etc.) for each variant. Column names are suffixed by the variant number.
        """
        data = {}

        # Get the number of samples from the first parameter to ensure consistency
        sample_length = len(self.results[next(iter(self.results))][0])  # Get the first variant's samples
        variant_names = [d.name if d.name else f'V{i + 1}' for i, d in enumerate(self.data)]

        # Iterate over each parameter in the results dictionary (e.g., theta, lam, etc.)
        for key in self.results.keys():
            for variant_idx, variant_samples in enumerate(self.results[key]):
                # Check if the sample length is consistent
                if len(variant_samples) != sample_length:
                    raise ValueError(f"All arrays must be of the same length. "
                                     f"Parameter {key} for variant {variant_idx + 1} has inconsistent length.")

                if len(self.results[key]) < len(variant_names):
                    column_name = f"{key}_{variant_names[variant_idx + 1]}"
                else:
                    column_name = f"{key}_{variant_names[variant_idx]}"

                # Add the samples for the variant as a column in the data dictionary
                data[column_name] = variant_samples

        df = pd.DataFrame(data)
        return df

    def summary(self):
        """Prints a summary of the results with probabilities, expected losses, and uplift."""
        data = []
        comparison_method = self.comparison_method

        # Determine variant names (use name attribute or default to V1, V2, etc.)
        variant_names = [d.name if d.name else f'V{i + 1}' for i, d in enumerate(self.data)]

        for variant_idx, variant_name in enumerate(variant_names):
            # Initialize row data for each variant
            row_data = {
                "Variant": variant_name,
                "Conversion Rate HDI 2.5%": None,
                "Conversion Rate Mean": None,
                "Conversion Rate HDI 97.5%": None,
                "Relative Uplift HDI 2.5%": None,
                "Relative Uplift Mean": None,
                "Relative Uplift HDI 97.5%": None,
                "Probability of Beating Control": None,
                "Expected Loss": None,
                "Expected Loss %": None
            }

            # Conversion Rate Metrics (theta)
            if 'Conversion Rate' in self.results:
                variant_samples = self.results['Conversion Rate'][variant_idx]

                # Calculate mean and 95% HDI for conversion rate
                mean_value = np.mean(variant_samples)
                hdi_2_5, hdi_97_5 = np.percentile(variant_samples, [2.5, 97.5])

                # Fetch conversion rate uplift
                if len(self.results['Relative Uplift']) < len(
                        variant_names):  # i.e. compare_to_control, control has no uplift column
                    idx = variant_idx - 1
                else:
                    idx = variant_idx

                uplift_samples = self.results['Relative Uplift'][idx] if idx >= 0 else None
                uplift_mean = np.mean(uplift_samples) if idx >= 0 else None
                uplift_hdi_2_5, uplift_hdi_97_5 = (
                    np.percentile(uplift_samples, [2.5, 97.5]) if idx >= 0 else (None, None))

                # Get probabilities, expected losses, and expected loss percentages
                probabilities, expected_losses, expected_losses_pct = analyze_results(self.results, 'Conversion Rate',
                                                                                      variant_names, comparison_method)

                # Handle `compare_to_control` comparison method
                if comparison_method == 'compare_to_control':
                    probability_of_winning = probabilities.get(f'{variant_name} vs {variant_names[0]}',
                                                               'NaN') if variant_idx > 0 else 'NaN'
                    expected_loss = expected_losses.get(f'{variant_name} vs {variant_names[0]}',
                                                        'NaN') if variant_idx > 0 else 'NaN'
                    expected_loss_pct = expected_losses_pct.get(f'{variant_name} vs {variant_names[0]}',
                                                                'NaN') if variant_idx > 0 else 'NaN'

                # Handle `best_of_rest` comparison method
                elif comparison_method == 'best_of_rest':
                    probability_of_winning = probabilities.get(f'{variant_name} vs Best of Rest', 'NaN')
                    expected_loss = expected_losses.get(f'{variant_name} vs Best of Rest', 'NaN')
                    expected_loss_pct = expected_losses_pct.get(f'{variant_name} vs Best of Rest', 'NaN')

                # Update row data for conversion rate
                row_data.update({
                    "Conversion Rate HDI 2.5%": hdi_2_5,
                    "Conversion Rate Mean": mean_value,
                    "Conversion Rate HDI 97.5%": hdi_97_5,
                    "Relative Uplift HDI 2.5%": uplift_hdi_2_5,
                    "Relative Uplift Mean": uplift_mean,
                    "Relative Uplift HDI 97.5%": uplift_hdi_97_5,
                    "Probability of Beating Control": probability_of_winning,
                    "Expected Loss": expected_loss,
                    "Expected Loss %": expected_loss_pct
                })

         
            # Append the row to the data list
            data.append(row_data)

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Columns for expected loss percentages and probabilities of winning
        percent_columns = [
            "Expected Loss %",
            "Conversion Rate HDI 2.5%",
            "Conversion Rate Mean",
            "Conversion Rate HDI 97.5%",
            "Probability of Beating Control",
            "Relative Uplift Mean",
            "Relative Uplift HDI 2.5%",
            "Relative Uplift HDI 97.5%"
        ]

        # Multiply by 100 and add '%' sign for both expected loss % and probability of winning
        for column in percent_columns:
            if column in df:
                df[column] = df[column].apply(
                    lambda x: f"{float(x) * 100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "NaN")

        return df
