import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="SimulatedAttackRUSSELL2000", page_icon="❗")

st.markdown("# SimulatedAttackRUSSELL2000")
st.sidebar.header("SimulatedAttackRUSSELL2000")

filename = 'new_russel.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'

pipeline = Pipeline(filename, start_date, end_date)
pipeline.run_pipeline()