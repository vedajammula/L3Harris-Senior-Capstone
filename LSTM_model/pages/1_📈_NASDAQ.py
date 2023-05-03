import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="NASDAQ", page_icon="📈")

st.markdown("# NASDAQ")
st.sidebar.header("NASDAQ")

filename = 'nasdaq_all.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'

pipeline = Pipeline(filename, start_date, end_date)
pipeline.run_pipeline()