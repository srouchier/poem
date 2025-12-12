import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

import arviz as az
from cmdstanpy import CmdStanModel

from utils import load_data_to_session

st.set_page_config(
    page_title="Linear regression",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
# Linear regression

This model is explained on [this page](https://buildingenergygeeks.com/linearregression_stan.html).

## Data selection

Load a dataset for model training.
""")

c01, c02 = st.columns(2)

with c01:
    st.subheader("Data file")

    # --- Upload widget + loader ----------------------------------------------
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
    if uploaded is not None:
        try:
            df = load_data_to_session(uploaded)
            st.success(f"Loaded `{uploaded.name}` — {len(df):,} rows")
            st.write("Displaying the first 5 lines of the dataset")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    elif "df" in st.session_state:
        df = st.session_state["df"]
        st.info("Using previously uploaded dataset")
        st.write("Displaying the first 5 lines of the dataset")
        st.dataframe(st.session_state["df"].head())

    # Attendre que l'utilisateur ait chargé des données pour continuer
    if "df" not in st.session_state:
        st.stop()

# ------------------- Scatter plot selector & display -------------------------
def show_scatter_plot(df: pd.DataFrame):
    if df.shape[1] < 2:
        st.info("Need at least two columns to plot.")
        return

    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns available for Y axis.")
        return

    c11, c12 = st.columns(2)
    x_col = c11.selectbox("X axis (any column)", cols, index=0)
    y_col = c12.selectbox("Y axis (numeric only)", numeric_cols, index=1)

    # Offer a simple checkbox to let the user opt into parsing X as datetime
    parse_dt = st.checkbox(f"Parse '{x_col}' as datetime?", value=False)
    if parse_dt:
        x_dt = pd.to_datetime(df[x_col], errors="coerce")
        n_valid = int(x_dt.notna().sum())
        if n_valid == 0:
            st.warning(f"No valid datetimes found in '{x_col}'.")
        else:
            plot_df = df.copy()
            plot_df.index = x_dt
            st.line_chart(plot_df[y_col].sort_index())
            return
    # numeric X -> set as index
    elif pd.api.types.is_numeric_dtype(df[x_col]):
        try:
            st.scatter_chart(df.set_index(x_col)[y_col], x_label=x_col, y_label=y_col)
            return
        except Exception:
            pass
    # fallback: group by categorical X (mean) then plot
    else:
        try:
            grouped = df.groupby(x_col)[y_col].mean()
            st.scatter_chart(grouped)
        except Exception as e:
            pass # st.error(f"Could not create plot: {e}")


with c02:
    st.subheader("Visualization")
    # Selecting X and Y variables
    if "df" in st.session_state:

        st.write("You may select two columns of the dataset to show a scatter plot. This may help you select input variables of the linear regression model.")
        show_scatter_plot(st.session_state["df"])

st.markdown("---")

# ------------------- MODELLING -------------------------

st.markdown("""
## Linear regression model

This model takes one variable as output (for example energy consumption),
and one or several variables as input. You may select them below.
""")

st.latex(r'''y_i = \alpha + \beta_1 x_{i1} + ... + \beta_k x_{ik} + \varepsilon''')

# two-column layout: left = X selection (checkboxes), right = Y selection (radio)
c1, c2 = st.columns(2, border=True, width=500)
cols = df.columns.tolist()

with c1:
    st.markdown("**Select input variables X**")

    # Persistent checkbox list in session
    prev_selection = st.session_state.get("X_cols", [])
    selected_cols = []
    for col in cols:
        if st.checkbox(col, value=(col in prev_selection), key=f"select_x_{col}"):
            selected_cols.append(col)

    st.session_state["X_cols"] = selected_cols
    st.session_state["X"] = df[selected_cols].copy()

with c2:
    st.markdown("**Select output variable Y**")
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    y_options = numeric_cols if numeric_cols else cols
    if not y_options:
        st.info("No columns available to select as Y. Make sure your target variable is a numeric type.")
    else:
        prev_y = st.session_state.get("y_col")
        default_index = y_options.index(prev_y) if (prev_y in y_options) else 0
        y_choice = st.radio("Select one variable", options=y_options, index=default_index, key="select_y_radio")
        st.session_state["y_col"] = y_choice
        st.session_state["Y"] = df[y_choice].copy()

# ------------------- TRAINING -------------------------

training_done = False

# Show train button only when at least one input column was selected
if ("Y" in st.session_state) and st.session_state.get("X_cols"):

    xcols = st.session_state["X_cols"]

    if st.button(f"Train model with {len(xcols)} inputs: {', '.join(xcols)}"):

        try:
            # training logic goes here
            model = CmdStanModel(stan_file='models_stan/linearregression.stan')

            model_data = {
                "N": len(df),
                "K": len(selected_cols),
                "x": st.session_state["X"].values,
                "y": st.session_state["Y"].values,
            }

            fit = model.sample(data=model_data)

            st.success("Model training complete")

            # save fit in session state for later use
            training_done = True
            st.session_state["fit"] = fit

        except Exception as e:
            st.error({e})
            st.stop()

else:
    st.warning("Please select at least one input variable (X) and an output variable (Y) to enable training.")
    st.stop()

st.markdown("---")

# ------------------- RESULTS -------------------------

# Only proceed if training has succeeded
if not training_done:
    st.stop()

st.markdown("## Results")

c21, c22 = st.columns(2)

# Show convergence diagnostics
with c21:

    st.markdown("**MCMC convergence diagnostics**")
    st.write(fit.diagnose())

    # Build a parameter list excluding diagnostics and predictions
    exclude_exact = {"lp__"}
    exclude_prefixes = ("log_lik", "y_hat")
    params = [p for p in fit.column_names if p not in exclude_exact and not any(p.startswith(pref) for pref in exclude_prefixes)]

    # Show summary table
    st.markdown("**Posterior summary**")
    summary_df = fit.summary(percentiles=(5, 50, 95))
    summary_df = summary_df.loc[[r for r in summary_df.index if r in params]]
    st.dataframe(summary_df)

# Save results
res_pd = fit.draws_pd()
res_xr = fit.draws_xr()

idata = az.from_cmdstanpy(
    posterior = fit,
    posterior_predictive="y_hat",
    observed_data={"y": st.session_state["Y"]},
    constant_data={"x": st.session_state["X"]},
    log_likelihood="log_lik",
)

with c22:
    
    # Show trace plot
    st.markdown("**Trace plot**")
    az.plot_trace(idata)  # choose vars you want
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)


st.markdown("## Posterior predictive plots")

y_post = idata.posterior_predictive["y_hat"]
# mean over chain and draw -> gives a DataArray indexed by observation
y_post_mean = y_post.mean(("chain", "draw"))

# Prepare observed y values
obs_name = st.session_state.get("y_col", "y")
y_obs = st.session_state.get("Y")
if y_obs is None:
    y_obs = df[obs_name]

# Get selected input variables (X)
input_vars = st.session_state.get("X_cols", [])
if not input_vars:
    st.info("No input variables available for posterior predictive plots.")
else:

    ncols = len(input_vars)
    fig_width = 5 * max(1, ncols)   # inches per column
    fig_height = 4                  # inches
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height), squeeze=False, dpi=150)

    for i, col in enumerate(input_vars):
        ax = axes[0, i]

        x = df[col]
        # scatter observed data
        ax.scatter(x, y_obs, alpha=0.6, label='data')
        # scatter posterior mean (align by observation index)
        ax.scatter(x, y_post_mean, c="C1", alpha=0.6, label='posterior mean')
        # plot HDI along x (az.plot_hdi accepts x and y_draws)
        az.plot_hdi(x, y_post, ax=ax)

        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel(obs_name, fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=10)

    plt.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)