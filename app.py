import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title('Settings')
    show_indicators = st.checkbox('Show Technical Indicators', value=True)
    prediction_days = st.slider('Prediction Days', 1, 30, 7)
    risk_tolerance = st.select_slider(
        'Risk Tolerance',
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate'
    )

# Title
st.title('Stock Trend Prediction')

# Add this after the title
st.markdown("""
### What is this app?
This app helps you understand if a stock might be a good investment. It looks at past prices and predicts future trends.
Simply enter a company's stock symbol (like AAPL for Apple, MSFT for Microsoft) to get started.
""")

# User input and date range
col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
with col2:
    start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
with col3:
    end_date = st.date_input('End Date', pd.to_datetime('2025-01-01'))

# Technical Indicators Functions
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def generate_signals(predictions, current_price, risk_tolerance):
    if risk_tolerance == 'Conservative':
        threshold = 0.02  # 2% change
    elif risk_tolerance == 'Moderate':
        threshold = 0.05  # 5% change
    else:  # Aggressive
        threshold = 0.08  # 8% change
    
    # Get the last prediction
    last_prediction = float(predictions[-1][0])  # Convert to float
    predicted_change = (last_prediction - current_price) / current_price
    
    if predicted_change > threshold:
        return "BUY", predicted_change
    elif predicted_change < -threshold:
        return "SELL", predicted_change
    else:
        return "HOLD", predicted_change

try:
    # Load the model
    with st.spinner('Loading model...'):
        model = load_model('model.keras')
        st.success("Model loaded successfully!")

    # Download data
    with st.spinner('Loading data...'):
        df = yf.download(user_input, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for {user_input}. Please try a different ticker.")
        else:
            # Calculate technical indicators
            df['MA100'] = df.Close.rolling(100).mean()
            df['MA200'] = df.Close.rolling(200).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['Signal'] = calculate_macd(df['Close'])

            # Company Information
            stock = yf.Ticker(user_input)
            info = stock.info

            st.subheader('Company Information')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.2f}")
            with col2:
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}")
            with col3:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
            with col4:
                st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")

            # Add this after company information
            st.markdown("""
            ### Company Overview
            This section shows basic information about the company:
            - **Market Cap**: The total value of the company
            - **P/E Ratio**: How expensive the stock is compared to company earnings
            - **52 Week High/Low**: The highest and lowest prices in the last year
            """)

            # Data Statistics
            st.subheader('Data Statistics')
            st.write(df.describe())

            # Price Chart with Moving Averages
            st.subheader('Price and Moving Averages')
            fig1 = plt.figure(figsize=(6,3))
            plt.plot(df.Close, label='Close Price', linewidth=1)
            plt.plot(df.MA100, 'r', label='MA100', linewidth=1)
            plt.plot(df.MA200, 'g', label='MA200', linewidth=1)
            plt.title(f'{user_input} Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(fig1)

            # Add this after the price chart
            st.markdown("""
            ### Price Analysis
            The chart above shows:
            - **Blue line**: Actual stock price over time
            - **Red line**: 100-day average price (helps spot trends)
            - **Green line**: 200-day average price (long-term trend)
            """)

            # Technical Indicators
            if show_indicators:
                st.subheader('Technical Indicators')
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    fig_rsi = plt.figure(figsize=(6,3))
                    plt.plot(df['RSI'], label='RSI')
                    plt.axhline(y=70, color='r', linestyle='--')
                    plt.axhline(y=30, color='g', linestyle='--')
                    plt.title('RSI')
                    plt.legend(fontsize=8)
                    plt.grid(True)
                    st.pyplot(fig_rsi)

                with col2:
                    # MACD
                    fig_macd = plt.figure(figsize=(6,3))
                    plt.plot(df['MACD'], label='MACD')
                    plt.plot(df['Signal'], label='Signal')
                    plt.title('MACD')
                    plt.legend(fontsize=8)
                    plt.grid(True)
                    st.pyplot(fig_macd)

            # Add this after technical indicators
            st.markdown("""
            ### Technical Analysis
            - **RSI (Relative Strength Index)**:
              - Above 70: Stock might be overpriced
              - Below 30: Stock might be underpriced
              - Between 30-70: Normal price range

            - **MACD (Moving Average Convergence Divergence)**:
              - Shows if the stock is trending up or down
              - Helps identify potential buy/sell points
            """)

            # Prepare data for prediction
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

            if len(data_training) > 0 and len(data_testing) > 0:
                # Scale the data
                scaler = MinMaxScaler(feature_range=(0,1))
                data_training_array = scaler.fit_transform(data_training)

                # Prepare sequences
                x_train = []
                y_train = []

                for i in range(100, data_training_array.shape[0]):
                    x_train.append(data_training_array[i-100: i])
                    y_train.append(data_training_array[i, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)

                # Make predictions
                with st.spinner('Making predictions...'):
                    past_100_days = data_training.tail(100)
                    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                    input_data = scaler.fit_transform(final_df)

                    x_test = []
                    y_test = []

                    for i in range(100, input_data.shape[0]):
                        x_test.append(input_data[i-100: i])
                        y_test.append(input_data[i, 0])

                    x_test, y_test = np.array(x_test), np.array(y_test)
                    y_predicted = model.predict(x_test)

                    # Inverse transform
                    scale_factor = 1/scaler.scale_[0]
                    y_predicted = y_predicted * scale_factor
                    y_test = y_test * scale_factor

                    # Get current price
                    current_price = float(df['Close'].iloc[-1])

                    # Generate trading signals
                    signal, predicted_change = generate_signals(y_predicted, current_price, risk_tolerance)

                    # Plot predictions
                    st.subheader('Predictions vs Original')
                    fig2 = plt.figure(figsize=(6,3))
                    plt.plot(y_test, 'b', label='Original Price', linewidth=1)
                    plt.plot(y_predicted, 'r', label='Predicted Price', linewidth=1)
                    plt.title(f'{user_input} Price Prediction')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend(fontsize=8)
                    plt.grid(True)
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                    st.pyplot(fig2)

                    # Calculate metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    import math

                    rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
                    mae = mean_absolute_error(y_test, y_predicted)
                    confidence = 100 - (rmse / current_price * 100)

                    # Display metrics and predictions
                    st.subheader('Prediction Summary')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Predicted Change", f"{predicted_change*100:.2f}%")
                    with col2:
                        st.metric("Root Mean Square Error", f"${rmse:.2f}")
                        st.metric("Mean Absolute Error", f"${mae:.2f}")
                    with col3:
                        st.metric("Prediction Confidence", f"{confidence:.2f}%")
                        st.metric("Risk Level", risk_tolerance)

                    # Trading Signal
                    st.subheader('Trading Signal')
                    if signal == "BUY":
                        st.success(f"🟢 {signal} Signal - Predicted price increase of {predicted_change*100:.2f}%")
                    elif signal == "SELL":
                        st.error(f"🔴 {signal} Signal - Predicted price decrease of {abs(predicted_change*100):.2f}%")
                    else:
                        st.warning(f"🟡 {signal} Signal - Predicted price change of {predicted_change*100:.2f}%")

                    # Add this after trading signal
                    st.markdown("""
                    ### Trading Signal Explained
                    - **🟢 BUY**: The stock price is expected to go up
                    - **🔴 SELL**: The stock price is expected to go down
                    - **🟡 HOLD**: The stock price is expected to stay stable

                    *Note: These are predictions, not financial advice. Always do your own research.*
                    """)

                    # Risk Assessment
                    st.subheader('Risk Assessment')
                    
                    # Calculate risk factors
                    volatility = float(df['Close'].pct_change().std() * 100)
                    rsi_value = float(df['RSI'].iloc[-1])
                    macd_value = float(df['MACD'].iloc[-1])
                    signal_value = float(df['Signal'].iloc[-1])
                    ma200_value = float(df['MA200'].iloc[-1])

                    # Determine RSI status
                    if rsi_value > 70:
                        rsi_status = 'Overbought'
                    elif rsi_value < 30:
                        rsi_status = 'Oversold'
                    else:
                        rsi_status = 'Neutral'

                    # Determine MACD signal
                    if macd_value > signal_value:
                        macd_signal = 'Bullish'
                    else:
                        macd_signal = 'Bearish'

                    # Determine price vs MA200
                    if current_price > ma200_value:
                        price_vs_ma = 'Above'
                    else:
                        price_vs_ma = 'Below'

                    # Display risk factors
                    risk_factors = {
                        'Market Volatility': f"{volatility:.2f}%",
                        'RSI Status': rsi_status,
                        'MACD Signal': macd_signal,
                        'Price vs MA200': price_vs_ma
                    }
                    
                    for factor, value in risk_factors.items():
                        st.write(f"{factor}: {value}")

                    # Add this after risk assessment
                    st.markdown("""
                    ### Risk Assessment Explained
                    - **Market Volatility**: How much the price typically moves
                    - **RSI Status**: Whether the stock is overpriced or underpriced
                    - **MACD Signal**: Current trend direction
                    - **Price vs MA200**: Whether the stock is above or below its long-term average

                    *Higher risk means bigger potential gains but also bigger potential losses.*
                    """)

            else:
                st.error("Not enough data for training and testing. Please try a different ticker.")

            # Add this at the end
            st.markdown("""
            ### How to Use This Information
            1. **Start with Company Overview**: Understand the company's size and value
            2. **Check Price Analysis**: See how the stock has performed
            3. **Look at Predictions**: Understand potential future movements
            4. **Consider Risk**: Make sure the risk level matches your comfort
            5. **Review Trading Signal**: Get a clear buy/sell/hold recommendation

            *Remember: Past performance doesn't guarantee future results. Always invest responsibly.*
            """)

except Exception as e:
    if "YFinance" in str(e):
        st.error("Error fetching data. Please try again later or use a different ticker.")
        st.info("This might be due to API rate limits or invalid ticker symbol.")
    else:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try a different stock ticker. For example: AAPL, MSFT, GOOGL, etc.")