import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import beta
from bayestest.ab_test import BayesTest

# App Configuration
st.set_page_config(page_title="Bayesian A/B Testing Calculator", layout="wide")
st.title("Bayesian A/B Testing Calculator")
st.write("Enter prior parameters and variant conversion results, and get posteriors and probability of winning. Work in progress.")

# Initialize session state
if 'test' not in st.session_state:
    st.session_state.test = BayesTest()
if 'prior_plot_ready' not in st.session_state:
    st.session_state.prior_plot_ready = False

# Sidebar for configuration
st.sidebar.header("Configuration")

# Priors with sliders
st.sidebar.subheader("Beta Prior Parameters")
alpha_prior = st.sidebar.slider("Alpha prior (shape parameter)", min_value=1, max_value=50, value=5, step=1)
beta_prior = st.sidebar.slider("Beta prior (shape parameter)", min_value=1, max_value=50, value=5, step=1)

# Compute the Beta prior plot only once
if st.sidebar.button("Select and View Priors"):
    st.session_state.prior_plot_ready = True

if st.session_state.prior_plot_ready:
    x_range = np.linspace(0, 1, 5_000)
    y = beta.pdf(x_range, alpha_prior, beta_prior)
    prior_chart = alt.Chart(pd.DataFrame({'x': x_range, 'Density': y})).mark_line().encode(
        x='x',
        y='Density'
    ).properties(
        title="Beta Prior Distribution",
        width=400,
        height=300
    )
    st.sidebar.subheader("Beta Distribution")
    st.sidebar.altair_chart(prior_chart, use_container_width=True)

# Variant inputs
st.sidebar.subheader("Variants")
num_variants = st.sidebar.number_input("Number of variants", min_value=1, value=1, step=1)
variants_data = []

for i in range(num_variants):
    st.sidebar.subheader(f"{'Control' if i == 0 else f'Variant {i}'}")
    name = st.sidebar.text_input(f"Name for {'Control' if i == 0 else f'Variant {i}'}", value=f"{'Control' if i == 0 else f'V{i}'}", key=f"name_{i}")
    visitors = st.sidebar.number_input(f"Visitors for {name}", min_value=1, value=1000, step=1, key=f"visitors_{i}")
    conversions = st.sidebar.number_input(f"Conversions for {name}", min_value=0, max_value=visitors, value=100, step=1, key=f"conversions_{i}")
    variants_data.append({'name': name, 'visitors': visitors, 'conversions': conversions, 'is_control': i == 0})

# Run the Bayesian test
if st.sidebar.button("Run Test"):
    # Set priors
    st.session_state.test.set_conversion_rate_prior(alpha_prior, beta_prior)

    # Reset and configure test
    st.session_state.test.reset()
    for variant in variants_data:
        st.session_state.test.add_variant(
            visitors=variant['visitors'],
            conversions=variant['conversions'],
            name=variant['name'],
            control=variant['is_control']
        )

    # Run the Bayesian test
    st.session_state.test.run(samples=5_000)
    df = st.session_state.test.posterior_samples()
    df_results = st.session_state.test.summary()

    # Function to prepare data for density plots
    def prepare_long_format(df, columns, value_name):
        return df[columns].reset_index().melt(
            id_vars=['index'], var_name='Variant', value_name=value_name
        ).rename(columns={'index': 'Sample'})

    # Extract columns dynamically
    theta_cols = [col for col in df.columns if col.startswith('theta_') and not col.startswith('theta_uplift_') and not col.startswith('theta_reluplift_')]
    uplift_cols = [col for col in df.columns if col.startswith('theta_uplift_')]
    rel_uplift_cols = [col for col in df.columns if col.startswith('theta_reluplift_')]

    # Prepare data for plotting
    theta_long = prepare_long_format(df, theta_cols, 'Conversion Rate')
    uplift_long = prepare_long_format(df, uplift_cols, 'Uplift')
    rel_uplift_long = prepare_long_format(df, rel_uplift_cols, 'Relative Uplift')

    # Function to plot density
    def plot_density(data, value_column, title):
        chart = alt.Chart(data).transform_density(
            value_column,
            as_=['Value', 'Density'],
            groupby=['Variant']
        ).mark_line().encode(
            x='Value:Q',
            y='Density:Q',
            color='Variant:N'
        ).properties(
            width=800,
            height=400,
            title=title
        )
        return chart

    # Display results
    st.subheader("Posterior Distributions")
    st.altair_chart(plot_density(theta_long, 'Conversion Rate', "Posterior Conversion Rate Distributions"), use_container_width=True)
    st.write("### Conversion Rate Summary")
    st.dataframe(df_results[['Variant', 'Conversion Rate Mean', 'Conversion Rate HDI 2.5%', 'Conversion Rate HDI 97.5%']])

    st.altair_chart(plot_density(uplift_long, 'Uplift', "Posterior Uplift Distributions"), use_container_width=True)
    st.write("### Uplift Summary")
    st.dataframe(df_results[['Variant', 'Conversion Rate Relative Uplift Mean', 'Conversion Rate Relative Uplift HDI 2.5%', 'Conversion Rate Relative Uplift HDI 97.5%']])

    st.altair_chart(plot_density(rel_uplift_long, 'Relative Uplift', "Posterior Relative Uplift Distributions"), use_container_width=True)
    st.write("### Probability of Winning and Expected Loss")
    st.dataframe(df_results[['Variant', 'Probability of Winning (Conversion Rate)', 'Expected Loss (Conversion Rate)', 'Expected Loss % (Conversion Rate)']])
