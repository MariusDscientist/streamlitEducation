import yfinance as yf
import pandas as pd
import streamlit as st

st.write("""
## Simple Stock Price App
         
Shown are the stock closing price and volume of google
         
""")
# Define the ticker symbol
tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
TICKERdF = tickerData.history(period='1d',
                              start='2010-5-31',
                              end='2020-5-31')
st.line_chart(TICKERdF['Close'])
st.line_chart(TICKERdF['Volume'])
