import streamlit as st
import pandas as pd

def load_data_to_session(uploaded_file):
    """Read uploaded file and store dataframe and filename in st.session_state."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.warning(f"File type not accepted")

    st.session_state["df"] = df
    st.session_state["uploaded_name"] = uploaded_file.name
    return df

def show_dataset_sidebar_note():
    """Show a small note in the sidebar when a dataset was uploaded."""
    df = st.session_state.get("df")
    name = st.session_state.get("uploaded_name")
    if df is None:
        return  # nothing to show
    try:
        rows, cols = df.shape
        info = f"{rows:,} rows × {cols} cols"
    except Exception:
        info = ""
    label = f"Dataset loaded{': ' + name if name else ''}"
    st.sidebar.markdown(f"**{label}**  \n{info}")