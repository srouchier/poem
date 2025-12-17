import streamlit as st
#import cmdstanpy
from utils import show_dataset_sidebar_note

pg = st.navigation([st.Page("streamlit/1_linreg.py", title="Linear regression", icon="📏")])
#                    st.Page("streamlit/0_index.py", title="Welcome", icon="🏠"),
#                    st.Page("streamlit/1_linreg.py", title="Linear regression", icon="📏"),
#                    st.Page("streamlit/2_changepoint.py", title="Change-point", icon="✍️"),
#                    st.Page("streamlit/3_timeseries.py", title="Time series", icon="🕒")])

# show_dataset_sidebar_note()

with st.sidebar:
    st.write("This sidebar will be used to list trained models so the user can load and compare them. Models trained in this session, or loaded from a previous session.")

# Attempt install
#if "cmdstan_installed" not in st.session_state:
#    try:
#        with st.spinner("Checking/Installing CmdStan..."):
#            cmdstanpy.install_cmdstan()
#        st.session_state["cmdstan_installed"] = True
#    except Exception as e:
#        st.session_state["cmdstan_installed"] = False
#        st.warning("CmdStan install failed")
#        st.exception(e)

pg.run()