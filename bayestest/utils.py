import numpy as np


def compute_probability_of_winning(results: dict, param_name: str, variant_names: list, comparison_method: str):
    probabilities = {}

    for i in range(len(variant_names)):
        variant_samples = np.array(results[param_name][i])

        if comparison_method == 'compare_to_control':
            # Compare to control (first variant is the control)
            control_samples = np.array(results[param_name][0])
            prob_win = np.mean(variant_samples > control_samples)
            probabilities[f'{variant_names[i]} vs {variant_names[0]}'] = prob_win

        elif comparison_method == 'best_of_rest':
            # Compare each variant to the best of the rest (excluding itself)
            other_samples = np.array([results[param_name][j] for j in range(len(variant_names)) if j != i])
            best_of_rest = np.max(other_samples, axis=0)
            prob_win = np.mean(variant_samples > best_of_rest)
            probabilities[f'{variant_names[i]} vs Best of Rest'] = prob_win

    return probabilities


def compute_expected_loss(results: dict, param_name: str, variant_names: list, comparison_method: str):
    expected_losses = {}

    for i in range(len(variant_names)):
        variant_samples = np.array(results[param_name][i])

        if comparison_method == 'compare_to_control':
            # Compare to control (first variant is the control)
            control_samples = np.array(results[param_name][0])
            loss = control_samples - variant_samples
            loss = loss[loss > 0]  # Only consider positive losses
            expected_loss = np.mean(loss) if len(loss) > 0 else 0
            expected_losses[f'{variant_names[i]} vs {variant_names[0]}'] = expected_loss

        elif comparison_method == 'best_of_rest':
            # Compare each variant to the best of the rest (excluding itself)
            other_samples = np.array([results[param_name][j] for j in range(len(variant_names)) if j != i])
            best_of_rest = np.max(other_samples, axis=0)
            loss = best_of_rest - variant_samples
            loss = loss[loss > 0]  # Only consider positive losses
            expected_loss = np.mean(loss) if len(loss) > 0 else 0
            expected_losses[f'{variant_names[i]} vs Best of Rest'] = expected_loss

    return expected_losses


def compute_expected_loss_percentage(results: dict, param_name: str, variant_names: list, comparison_method: str):
    expected_losses_pct = {}

    for i in range(len(variant_names)):
        variant_samples = np.array(results[param_name][i])

        if comparison_method == 'compare_to_control':
            # Compare to control (first variant is the control)
            control_samples = np.array(results[param_name][0])

            # Calculate the percentage loss where the control is better than the variant
            pct_loss = (control_samples - variant_samples) / control_samples
            pct_loss = pct_loss[pct_loss > 0]  # Only consider positive percentage losses
            expected_loss_pct = np.mean(pct_loss) if len(pct_loss) > 0 else 0
            expected_losses_pct[f'{variant_names[i]} vs {variant_names[0]}'] = expected_loss_pct

        elif comparison_method == 'best_of_rest':
            # Compare each variant to the best of the rest (excluding itself)
            other_samples = np.array([results[param_name][j] for j in range(len(variant_names)) if j != i])
            best_of_rest = np.max(other_samples, axis=0)

            # Calculate the percentage loss where the best of rest is better than the variant
            pct_loss = (best_of_rest - variant_samples) / best_of_rest
            pct_loss = pct_loss[pct_loss > 0]  # Only consider positive percentage losses
            expected_loss_pct = np.mean(pct_loss) if len(pct_loss) > 0 else 0
            expected_losses_pct[f'{variant_names[i]} vs Best of Rest'] = expected_loss_pct

    return expected_losses_pct


def analyze_results(results: dict, param_name: str, variant_names: list, comparison_method: str):
    probabilities = compute_probability_of_winning(results, param_name, variant_names, comparison_method)
    expected_losses = compute_expected_loss(results, param_name, variant_names, comparison_method)
    expected_losses_pct = compute_expected_loss_percentage(results, param_name, variant_names, comparison_method)

    return probabilities, expected_losses, expected_losses_pct
