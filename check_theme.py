import streamlit as st
st.write("Theme base:", st.get_option("theme.base"))
st.write("BG:", st.get_option("theme.backgroundColor"))
st.write("Secondary BG:", st.get_option("theme.secondaryBackgroundColor"))
st.write("Text:", st.get_option("theme.textColor"))
