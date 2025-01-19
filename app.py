import streamlit as st
import pandas as pd
import plotly.express as px
from bayestest.ab_test import BayesTest

st.set_page_config(page_title="Bayesian A/B Testing", layout="wide")
st.title("Bayesian A/B Testing")

# Initialize session state
if 'test' not in st.session_state:
    st.session_state.test = BayesTest()
if 'num_variants' not in st.session_state:
    st.session_state.num_variants = 1
if 'alpha_prior' not in st.session_state:
    st.session_state.alpha_prior = 5
if 'beta_prior' not in st.session_state:
    st.session_state.beta_prior = 5

# Sidebar for adding variants
st.sidebar.header("Add Variants")
num_variants = st.sidebar.number_input("Number of variants", min_value=1, value=st.session_state.num_variants)
st.session_state.num_variants = num_variants
alpha_prior = st.sidebar.number_input("Alpha prior", min_value=1, value=st.session_state.alpha_prior)
st.session_state.alpha_prior = alpha_prior
beta_prior = st.sidebar.number_input("Beta prior", min_value=1, value=st.session_state.beta_prior)
st.session_state.beta_prior = beta_prior

st.session_state.test.set_conversion_rate_prior(st.session_state.alpha_prior, st.session_state.beta_prior)

variants_data = []
control_added = False

# Input fields for each variant
for i in range(num_variants):
    st.sidebar.subheader(f"{'Control' if i == 0 else f'Variant {i}'}")
    
    name = st.sidebar.text_input(f"Name for {'Control' if i == 0 else f'Variant {i}'}", 
                                value=f"{'Control' if i == 0 else f'V{i}'}", 
                                key=f"name_{i}")
    visitors = st.sidebar.number_input(f"Visitors for {name}", 
                                     min_value=1, 
                                     value=1000, 
                                     key=f"visitors_{i}")
    conversions = st.sidebar.number_input(f"Conversions for {name}", 
                                        min_value=0, 
                                        max_value=visitors,
                                        value=100, 
                                        key=f"conversions_{i}")
    
    variants_data.append({
        'name': name,
        'visitors': visitors,
        'conversions': conversions,
        'is_control': i == 0
    })

# Run test button
if st.sidebar.button("Run Test"):
    # Reset the test
    st.session_state.test.reset()
    
    # Add variants to the test
    for variant in variants_data:
        st.session_state.test.add_variant(
            visitors=variant['visitors'],
            conversions=variant['conversions'],
            name=variant['name'],
            control=variant['is_control']
        )
    
    # Run the test
    st.session_state.test.run(samples=50_000)
    
    # Get results
    df = st.session_state.test.posterior_samples()
    
    # Display results
    st.header("Test Results")


    import streamlit as st
    import pandas as pd
    import altair as alt

    # Load your DataFrame (replace this with your actual DataFrame loading logic)
    # Example:
    # df = pd.read_csv("your_file.csv")

    # Dynamic filtering for columns
    theta_cols = [col for col in df.columns if col.startswith('theta_') and not col.startswith('theta_uplift_') and not col.startswith('theta_reluplift_')]
    uplift_cols = [col for col in df.columns if col.startswith('theta_uplift_')]
    rel_uplift_cols = [col for col in df.columns if col.startswith('theta_reluplift_')]

    # Function to prepare data for Altair plotting
    def prepare_long_format(df, columns, value_name):
        long_df = df[columns].reset_index().melt(
            id_vars=['index'], var_name='Variant', value_name=value_name
        )
        long_df.rename(columns={'index': 'Sample'}, inplace=True)
        return long_df

    # Prepare data in long format
    theta_long = prepare_long_format(df, theta_cols, 'Conversion Rate')
    uplift_long = prepare_long_format(df, uplift_cols, 'Uplift')
    rel_uplift_long = prepare_long_format(df, rel_uplift_cols, 'Relative Uplift')

    # Function to create density plots
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

    # Plot Conversion Rate Posterior Distributions
    st.subheader("Conversion Rate Posterior Distributions")
    conversion_chart = plot_density(theta_long, 'Conversion Rate', 'Posterior Conversion Rate Distributions')
    st.altair_chart(conversion_chart, use_container_width=True)

    # Plot Uplift Posterior Distributions
    st.subheader("Uplift Posterior Distributions")
    uplift_chart = plot_density(uplift_long, 'Uplift', 'Posterior Uplift Distributions')
    st.altair_chart(uplift_chart, use_container_width=True)

    # Plot Relative Uplift Posterior Distributions
    st.subheader("Relative Uplift Posterior Distributions")
    rel_uplift_chart = plot_density(rel_uplift_long, 'Relative Uplift', 'Posterior Relative Uplift Distributions')
    st.altair_chart(rel_uplift_chart, use_container_width=True)

