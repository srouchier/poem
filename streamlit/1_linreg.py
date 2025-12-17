import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import io

import arviz as az
import pymc as pm

from utils import load_data_to_session

st.set_page_config(
    page_title="Linear regression",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
# Linear regression

This is a first draft of a web app for probabilistic modelling using PyMC. In the future, this app will let users select a model among a few options for building energy modelling.

For now, only ordinary linear regression is available here. I made tutorials for Bayesian regression [with PyMC](https://buildingenergygeeks.com/linearregression.html) or [with Stan](https://buildingenergygeeks.com/linearregression_stan.html).

## Training data

Start by loading a training dataset here.
""")

c01, c02 = st.columns(2)

with c01:
    st.subheader("Data file")

    # --- Upload widget + loader ----------------------------------------------
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"], key="linreg_train")
    if uploaded is not None:
        try:
            df, uploaded_name = load_data_to_session(uploaded)
            st.success(f"Loaded `{uploaded.name}` — {len(df):,} rows")
            st.session_state["df"] = df
            st.session_state["df_name"] = uploaded_name
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
and one or several variables as input. You may select any of the variables from the column names of the datafile you provided, then click on the Train button.

In a future version of this app, this section will allow prior selection.
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
        st.session_state["Y_col"] = y_choice
        st.session_state["Y"] = df[y_choice].copy()

# Prior specification: for now it's fixed

a_mu_prior = st.session_state["Y"].mean()
a_sigma_prior = st.session_state["Y"].std()
b_mu_prior = np.zeros(st.session_state["X"].shape[1])
b_sigma_prior = st.session_state["Y"].mean() / st.session_state["X"].mean(axis=0)

# ------------------- TRAINING -------------------------


# Show train button only when at least one input column was selected
if ("Y" in st.session_state) and st.session_state.get("X_cols"):

    xcols = st.session_state["X_cols"]
    if st.button(f"Train model with {len(xcols)} inputs: {', '.join(xcols)}"):

        try:

            with st.spinner('🔄 Sampling... This may take several minutes.'):

                # If there was a previous training, remove its results
                for k in ("summary_df", "cached_trace_fig", "idata", "idata_2"):
                    st.session_state.pop(k, None)

                model1 = pm.Model()

                with model1:

                    pred = pm.Data("pred", st.session_state["X"])

                    # Priors for unknown model parameters
                    alpha = pm.Normal("alpha", mu = a_mu_prior, sigma = a_sigma_prior)
                    beta = pm.Normal("beta", mu = b_mu_prior, sigma = b_sigma_prior, shape = st.session_state["X"].shape[1])
                    sigma = pm.HalfNormal("sigma", sigma=a_sigma_prior)

                    mu = alpha + pm.math.dot(pred, beta)
                    
                    # Likelihood (sampling distribution) of observations
                    Y_obs = pm.Normal("Y_obs", mu = mu, sigma = sigma, observed=st.session_state["Y"])

                    # Training
                    idata = pm.sample(chains=4, idata_kwargs={"log_likelihood": True})

                    # Posterior
                    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
                

            st.success("Model training complete")

            # Save idata in session state for later use
            training_done = True
            st.session_state["idata"] = idata
            st.session_state['model1'] = model1

        except Exception as e:
            st.error({e})
            st.stop()

else:
    st.warning("Please select at least one input variable (X) and an output variable (Y) to enable training.")
    st.stop()

st.markdown("---")

# ------------------- RESULTS -------------------------

# Only proceed if training has succeeded
if "idata" not in st.session_state:
    st.stop()
else:
    idata = st.session_state["idata"]

# Posterior predictive
y_hat = idata.posterior_predictive["Y_obs"]
# mean over chain and draw -> gives a DataArray indexed by observation
y_hat_mean = y_hat.mean(("chain", "draw"))
# Mean residuals
resid = st.session_state["Y"] - y_hat_mean
ssres = np.sum(resid**2)
sstot = np.sum( (st.session_state["Y"]- st.session_state["Y"].mean())**2 )
R2 = 1 - ssres / sstot
CVRMSE = np.sqrt( ssres / len(st.session_state["Y"]) ) / st.session_state["Y"].mean()

st.markdown("## Results")

c21, c22 = st.columns(2)

# Show convergence diagnostics
with c21:

    # Show summary table
    st.markdown("**Posterior summary**")
    if "summary_df" not in st.session_state:
        st.session_state.summary_df = az.summary(idata)
    st.dataframe(st.session_state.summary_df)

    st.markdown("**Some statistics**")
    st.write("R2: ", R2)
    st.write("CV(RMSE): ", CVRMSE)

    # az.plot_autocorr(resid.values)
    # fig = plt.gcf()
    # st.pyplot(fig, clear_figure = True, width=400)

with c22:
    
    # Show trace plot
    st.markdown("**Trace plot**")

    if "cached_trace_fig" in st.session_state:
        st.image(st.session_state["cached_trace_fig"])
    else:
        az.plot_trace(idata)
        fig = plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.session_state["cached_trace_fig"] = buf.getvalue()
        st.image(st.session_state["cached_trace_fig"])

        #st.pyplot(fig_trace, clear_figure = True)
    #plt.close(fig)

# ------------------- PREDICTIONS -------------------------


st.markdown("## Predictions")

c31, c32 = st.columns(2)

with c31:

    st.subheader("In-sample posterior prediction")

    st.write("Comparison of the model's output with the training data. Most data points should belong in the high density intervals.")

    x_plot_choice = st.selectbox("Select an X axis", cols, index=0)
    x_plot = df[x_plot_choice]
    y_plot = df[y_choice]

    fig, ax = plt.subplots()
    ax.scatter(x_plot, y_plot, alpha=0.6, label='data')
    ax.scatter(x_plot, y_hat_mean, c="C1", alpha=0.6, label='posterior mean')
    az.plot_hdi(x_plot, y_hat, ax=ax)
    ax.set_xlabel(x_plot_choice) #, fontsize=10)
    ax.set_ylabel(y_choice) #, fontsize=10)
    # ax.tick_params(axis='both', labelsize=10)
    ax.legend()#fontsize=10)
    st.pyplot(fig)
    plt.close(fig)


with c32:

    st.subheader("Out of sample prediction")

    # --- Upload widget + loader ----------------------------------------------
    uploaded_post = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"], key="linreg_test")
    if uploaded_post is not None:
        try:
            df_post, uploaded_name = load_data_to_session(uploaded_post)
            st.success(f"Loaded `{uploaded_post.name}` — {len(df):,} rows")
            st.session_state["df_post"] = df_post
            st.session_state["df_post_name"] = uploaded_name
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    elif "df_post" in st.session_state:
        df_post = st.session_state["df_post"]
        st.info("Using previously uploaded dataset")
    else:
        st.stop()

    try:

        st.session_state["X_post"] = df_post[selected_cols].copy()
        st.session_state["Y_post"] = df_post[y_choice].copy()

        if "idata_2" in st.session_state:
            idata_2 = st.session_state["idata_2"]
        else:

            model1 = st.session_state['model1']

            with model1:
                # update values of predictors:
                pm.set_data({"pred": st.session_state["X_post"]})
                # use the updated values and predict outcomes and probabilities:
                idata_2 = pm.sample_posterior_predictive(
                    idata,
                    var_names=["Y_obs"],
                    return_inferencedata=True,
                    predictions=True,
                    extend_inferencedata=False
                )
            st.session_state["idata_2"] = idata_2

        # Posterior predictive
        y_post = idata_2.predictions["Y_obs"]
        # mean over chain and draw -> gives a DataArray indexed by observation
        y_post_mean = y_post.mean(("chain", "draw"))

        x_plot = df_post[x_plot_choice]
        y_plot = df_post[y_choice]
        fig, ax = plt.subplots()
        ax.scatter(x_plot, y_plot, alpha=0.6, label='data')
        ax.scatter(x_plot, y_post_mean, c="C1", alpha=0.6, label='posterior mean')
        az.plot_hdi(x_plot, y_post, ax=ax)
        ax.set_xlabel(x_plot_choice) #, fontsize=10)
        ax.set_ylabel(y_choice) #, fontsize=10)
        # ax.tick_params(axis='both', labelsize=10)
        ax.legend()#fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning("The prediction dataset must have the same column names as the training dataset.")
        st.error(f"Could not create plot: {e}")

