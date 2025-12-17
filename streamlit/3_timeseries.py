import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Time series models", page_icon="🕒")
st.markdown("""
# Time series models

This page will demonstrate ARMA models, mixed with linear regression or energy signatures. I will start adding these models after I have some feedback on the web app.
""")