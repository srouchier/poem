import streamlit as st

st.write("# Probabilistic building energy modelling")

# ajouter une image workflow et une image modèle

st.markdown(
    """
    This page is a first draft of a future web app for the POEM library for Probabilistic Building Energy Modelling. The applications will be:
    - On-going monitoring of energy consumption,
    - Measurement and Verification protocols (IPMVP options C and D)
    - Fault detection and diagnosis,
    - Performance assessment, energy audit, etc.

    The POEM library aims to guide energy analysts through a bayesian data analysis workflow: prior selection, MCMC convergence diagnosis, posterior prediction, model comparison, etc. For now, only a few models are available on the sidebar as examples, with limited functionalities.

    The library will be available as a Python package, or usable on this Streamlit app without any installation on the user's device. I also wrote a handbook on probabilistic modelling for building energy performance monitoring available [here](https://buildingenergygeeks.com/).

"""
)
