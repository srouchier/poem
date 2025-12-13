import streamlit as st
import cmdstanpy
from utils import show_dataset_sidebar_note

pg = st.navigation([st.Page("0_index.py", title="Welcome", icon="🏠"),
                    st.Page("1_linreg.py", title="Linear regression", icon="📏"),
                    st.Page("2_changepoint.py", title="Change-point", icon="✍️"),
                    st.Page("3_timeseries.py", title="Time series", icon="🕒")])

# show_dataset_sidebar_note()

with st.sidebar:
    st.write("I will probably use this sidebar to list trained models so the user can load and compare them. Models trained in this session, or loaded from a previous session.")

if "cmdstan_installed" not in st.session_state:
    cmdstanpy.install_cmdstan()
    st.session_state["cmdstan_installed"] = True

pg.run()