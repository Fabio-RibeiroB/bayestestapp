import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import beta
from bayestest.ab_test import BayesTest

# App Configuration
st.set_page_config(page_title="Bayesian A/B Testing", layout="wide")
st.title("Bayesian A/B Testing")

# Initialize session state
if 'test' not in st.session_state:
    st.session_state.test = BayesTest()
if 'num_variants' not in st.session_state:
    st.session_state.num_variants = 1
if 'show_prior_plot' not in st.session_state:
    st.session_state.show_prior_plot = False

# Sidebar for configuration
st.sidebar.header("Configuration")

# Priors with sliders
st.sidebar.subheader("Beta Prior Parameters")
alpha_prior = st.sidebar.slider("Alpha prior (shape parameter)", min_value=1, max_value=50, value=5, step=1)
beta_prior = st.sidebar.slider("Beta prior (shape parameter)", min_value=1, max_value=50, value=5, step=1)

# Button to show prior plot
if st.sidebar.button("Select and View Priors"):
    st.session_state.show_prior_plot = True

# Conditionally display the Beta prior plot
if st.session_state.show_prior_plot:
    st.sidebar.subheader("Beta Distribution")
    x_range = np.linspace(0, 1, 100_000)
    y = beta.pdf(x_range, alpha_prior, beta_prior)

    alt_prior_chart = alt.Chart(pd.DataFrame({'x': x_range, 'Density': y})).mark_line().encode(
        x='x',
        y='Density'
    ).properties(
        title="Beta Prior Distribution",
        width=400,
        height=300
    )
    st.sidebar.altair_chart(alt_prior_chart, use_container_width=True)

# Set priors in session state
st.session_state.test.set_conversion_rate_prior(alpha_prior, beta_prior)

# Add variants dynamically
num_variants = st.sidebar.number_input("Number of variants", min_value=1, value=st.session_state.num_variants)
st.session_state.num_variants = num_variants

variants_data = []
for i in range(num_variants):
    st.sidebar.subheader(f"{'Control' if i == 0 else f'Variant {i}'}")
    name = st.sidebar.text_input(f"Name for {'Control' if i == 0 else f'Variant {i}'}", value=f"{'Control' if i == 0 else f'V{i}'}", key=f"name_{i}")
    visitors = st.sidebar.number_input(f"Visitors for {name}", min_value=1, value=1000, key=f"visitors_{i}")
    conversions = st.sidebar.number_input(f"Conversions for {name}", min_value=0, max_value=visitors, value=100, key=f"conversions_{i}")
    variants_data.append({'name': name, 'visitors': visitors, 'conversions': conversions, 'is_control': i == 0})

# Run Test Button
if st.sidebar.button("Run Test"):
    # Reset test and add variants
    st.session_state.test.reset()
    for variant in variants_data:
        st.session_state.test.add_variant(
            visitors=variant['visitors'],
            conversions=variant['conversions'],
            name=variant['name'],
            control=variant['is_control']
        )
    
    # Run Bayesian Test
    st.session_state.test.run(samples=50_000)
    df = st.session_state.test.posterior_samples()

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

    # Plot posterior distributions
    st.subheader("Posterior Distributions")
    
    st.altair_chart(plot_density(theta_long, 'Conversion Rate', "Posterior Conversion Rate Distributions"), use_container_width=True)
    st.altair_chart(plot_density(uplift_long, 'Uplift', "Posterior Uplift Distributions"), use_container_width=True)
    st.altair_chart(plot_density(rel_uplift_long, 'Relative Uplift', "Posterior Relative Uplift Distributions"), use_container_width=True)
