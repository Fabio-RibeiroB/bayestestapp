import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
from scipy.stats import beta
from bayestest.ab_test import BayesTest

# Function to generate a scalable color palette
def get_scalable_color_palette(num_variants):
    return sns.color_palette("husl", num_variants).as_hex()

# App Configuration
st.set_page_config(page_title="Bayesian A/B Testing Calculator", layout="wide")
st.title("Bayesian A/B Testing Calculator")

# Initialize session state
if 'test' not in st.session_state:
    st.session_state.test = BayesTest()
if 'prior_plot_ready' not in st.session_state:
    st.session_state.prior_plot_ready = False

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.subheader("Beta Prior Parameters")
alpha_prior = st.sidebar.slider("Alpha prior", min_value=1, max_value=50, value=5, step=1)
beta_prior = st.sidebar.slider("Beta prior", min_value=1, max_value=50, value=5, step=1)

if st.sidebar.button("Select Prior"):
    st.session_state.prior_plot_ready = True
    st.sidebar.success("Prior confirmed ✅")

# Variant inputs
st.sidebar.subheader("Variants")
num_variants = st.sidebar.number_input("Number of variants", min_value=1, value=2, step=1)
variants_data = []

for i in range(num_variants):
    st.sidebar.subheader(f"{'Control' if i == 0 else f'Variant {i}'}")
    name = st.sidebar.text_input(f"Name for {'Control' if i == 0 else f'Variant {i}'}", value=f"{'Control' if i == 0 else f'V{i}'}", key=f"name_{i}")
    visitors = st.sidebar.number_input(f"Visitors for {name}", min_value=1, value=1000, step=1, key=f"visitors_{i}")
    conversions = st.sidebar.number_input(f"Conversions for {name}", min_value=0, max_value=visitors, value=100, step=1, key=f"conversions_{i}")
    variants_data.append({'name': name, 'visitors': visitors, 'conversions': conversions, 'is_control': i == 0})

# Run Bayesian test
if st.sidebar.button("Run Test"):
    st.session_state.test.set_conversion_rate_prior(alpha_prior, beta_prior)
    st.session_state.test.reset()

    for variant in variants_data:
        st.session_state.test.add_variant(
            visitors=variant['visitors'],
            conversions=variant['conversions'],
            name=variant['name'],
            control=variant['is_control']
        )

    st.session_state.test.run(samples=5000)
    df = st.session_state.test.posterior_samples()
    df_results = st.session_state.test.summary()

    prob_winning = df_results[['Variant', 'Probability of Beating Control']].drop(0)
    expected_loss = df_results[['Variant', 'Expected Loss']].drop(0)

    prob_winning['Probability of Beating Control'] = (
        prob_winning['Probability of Beating Control']
        .str.rstrip('%')
        .astype(float)
    )

    expected_loss['Expected Loss'] = (
        expected_loss['Expected Loss']
        .str.rstrip('%')
        .astype(float)
    )

    # Create consistent color mapping
    color_palette = get_scalable_color_palette(num_variants)
    variant_names = df_results['Variant'].unique()
    variant_color_dict = {variant: color_palette[i] for i, variant in enumerate(variant_names)}
    color_scale = alt.Scale(domain=list(variant_color_dict.keys()), range=list(variant_color_dict.values()))

    # **2x2 Layout**
    row1 = st.columns(2)
    row2 = st.columns(2)

    with row1[0]:  # Top Left - Probability of Winning
        st.subheader("Probability of Winning", help="The probability of beating the control")
        prob_winning_chart = alt.Chart(prob_winning).mark_bar().encode(
            x=alt.X('Variant:N', title="Variants"),
            y=alt.Y('Probability of Beating Control:Q', title="Probability of Beating Control (%)"),
            color=alt.Color('Variant:N', scale=color_scale)
        ).properties(width=300, height=300)
        st.altair_chart(prob_winning_chart, use_container_width=True)

    with row1[1]:  # Top Right - Expected Loss
        st.subheader("Expected Loss", help="The absolute decrease in conversion rate, compared to control, if you're wrong about the winner")
        expected_loss_chart = alt.Chart(expected_loss).mark_bar().encode(
            x=alt.X('Variant:N', title="Variants"),
            y=alt.Y('Expected Loss:Q', title="Absolute Expected Loss (%)"),
            color=alt.Color('Variant:N', scale=color_scale)
        ).properties(width=300, height=300)
        st.altair_chart(expected_loss_chart, use_container_width=True)

    # Data Preparation
    def prepare_long_format(df, columns, value_name):
        return df[columns].reset_index().melt(
            id_vars=['index'], var_name='Variant', value_name=value_name
        ).rename(columns={'index': 'Sample'})

    theta_cols = [col for col in df.columns if col.startswith('Conversion Rate') and not col.startswith('Uplift') and not col.startswith('Relative Uplift')]
    rel_uplift_cols = [col for col in df.columns if col.startswith('Relative Uplift')]

    theta_long = prepare_long_format(df, theta_cols, 'Conversion Rate')
    rel_uplift_long = prepare_long_format(df, rel_uplift_cols, 'Relative Uplift')

    # **Fix Variant Names**
    theta_long['Variant'] = theta_long['Variant'].str.replace(r'^Conversion Rate_', '', regex=True)
    rel_uplift_long['Variant'] = rel_uplift_long['Variant'].str.replace(r'^Relative Uplift_', '', regex=True)

    def plot_density(data, value_column, title):
        return alt.Chart(data).transform_density(
            value_column,
            as_=['Value', 'Density'],
            groupby=['Variant']
        ).mark_line().encode(
            x='Value:Q',
            y='Density:Q',
            color=alt.Color('Variant:N', scale=color_scale)
        ).properties(width=350, height=300, title=title)

    with row2[0]:  # Bottom Left - Posterior Distributions & Table
        st.subheader("Posterior Conversion Rate Distributions")
        st.altair_chart(plot_density(theta_long, 'Conversion Rate', "Posterior Conversion Rate Distributions"), use_container_width=True)
        st.dataframe(df_results[['Variant', 'Conversion Rate Mean', 'Conversion Rate HDI 2.5%', 'Conversion Rate HDI 97.5%']])

    with row2[1]:  # Bottom Right - Relative Uplift & Table
        st.subheader("Posterior Relative Uplift Distributions")
        st.altair_chart(plot_density(rel_uplift_long, 'Relative Uplift', "Posterior Relative Uplift Distributions"), use_container_width=True)
        st.dataframe(df_results[['Variant', 'Relative Uplift Mean', 'Relative Uplift HDI 2.5%', 'Relative Uplift HDI 97.5%']].drop(0))
