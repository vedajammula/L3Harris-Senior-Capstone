import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="RUSSELL 2000", page_icon="📈")

st.markdown("# RUSSELL 2000")
st.sidebar.header("RUSSELL 2000")

filename = 'russel2000_all.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'

pipeline = Pipeline(filename, start_date, end_date)
pipeline.run_pipeline()