import streamlit as st
import requests
from data_processing import *
import matplotlib.pyplot as plt
import ta
import yfinance as yf

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Title
st.title("📊 Stock Price Prediction(Deep Learning) Dashboard")

# Sidebar inputs for ticker, start date, and end date
selected_ticker = st.sidebar.selectbox("Select Ticker", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
start_date = st.sidebar.date_input("Select Start Date", min_value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Select End Date", min_value=start_date)
prediction_days = st.sidebar.slider("Number of Days to Predict", 5, 100, 50)

# Get company information from FMP API
FMP_API_KEY = "85HTy4OZQm3bnXdNDZnU8yOOUijKkbdY"
def get_company_info(ticker):
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    company_info = response.json()
    return company_info[0] if company_info else None

# Display company info if button is clicked
if st.sidebar.button("Get Information"):
    company_info = get_company_info(selected_ticker)
    data = fetch_stock_data(selected_ticker, start_date, end_date)
    train_df, val_df, test_df = preprocess_and_split_data(data)
    save_split_datasets(train_df, val_df, test_df)
    
    if company_info:
        # Save company information in session state
        st.session_state.company_info = company_info
        st.session_state.selected_ticker = selected_ticker
    else:
        st.error("Failed to fetch company information. Please try again.")

# Show company info in the center if it's available in session state
if 'company_info' in st.session_state:
    company_info = st.session_state.company_info
    selected_ticker = st.session_state.selected_ticker
    st.markdown(f"## 📈 {company_info['companyName']} ({selected_ticker})")
    st.image(company_info['image'], width=100)
    st.write(f"**Currency:** {company_info['currency']}")
    st.write(f"**Exchange:** {company_info['exchange']}")
    st.write(f"**Industry:** {company_info['industry']}")
    st.write(f"**CEO:** {company_info['ceo']}")
    st.write(f"**Changes:** {company_info['changes']}")
    st.write(f"**Market Cap:** {company_info['mktCap']}")
    st.write(f"**Volume:** {company_info['volAvg']}")
    st.write(f"**IPO Date:** {company_info['ipoDate']}")
    st.write(f"**Sector:** {company_info['sector']}")
    st.write(f"**Description:** {company_info['description']}")

    st.markdown("### 📊 Technical Indicators (on training data)")

    train_df, val_df, test_df = load_datasets()
    
    # Merge all dataframes for technical indicators
    complete_df = pd.concat([train_df, val_df, test_df], axis=0)
    complete_df = complete_df.sort_values('Date')  # Ensure data is sorted by date
    complete_df_with_indicators = add_technical_indicators(complete_df)

    # Plot 1: Close Price with SMA and EMA
    fig1, ax1 = plt.subplots(figsize=(18, 4))
    ax1.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["Close"], label="Close Price", color="blue")
    ax1.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["SMA_20"], label="SMA 20", color="orange")
    ax1.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["EMA_20"], label="EMA 20", color="green")
    ax1.set_title("Close Price with SMA and EMA")
    ax1.legend()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot 2: RSI
    fig2, ax2 = plt.subplots(figsize=(18, 2.5))
    ax2.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["RSI_14"], label="RSI 14", color="purple")
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")
    ax2.grid(True)
    st.pyplot(fig2)

    # Plot 3: MACD and Signal Line
    fig3, ax3 = plt.subplots(figsize=(18, 3))
    ax3.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["MACD"], label="MACD", color="blue")
    ax3.plot(complete_df_with_indicators["Date"], complete_df_with_indicators["Signal_Line"], label="Signal Line", color="orange")
    ax3.set_title("MACD and Signal Line")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)


# Button to load data and train model
# Button to load data, train model, and display predictions
if st.button("Load Data and Train Model"):
    with st.spinner("Loading datasets..."):
        train_df, val_df, test_df = load_datasets()
        st.success("Datasets loaded.")
        st.write("Preview of Training Data:")
        data = fetch_stock_data(selected_ticker, start_date, end_date)
        st.dataframe(data.head())

    # Process data
    data_train_scaled, data_val_scaled, data_test_scaled = extract_features(train_df, val_df, test_df)
    sequence_size = 60
    X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)

    data_all_scaled = np.concatenate([data_train_scaled, data_val_scaled, data_test_scaled], axis=0)
    train_size = len(data_train_scaled)
    validate_size = len(data_val_scaled)
    test_size = len(data_test_scaled)

    X_val, y_val = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)
    X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    with st.spinner("Training model..."):
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64)
        st.success("Model training complete.")

    # Plot training history
    st.line_chart({
        "Training Loss": history.history['loss'],
        "Validation Loss": history.history['val_loss']
    })

    # Evaluate on test data
    rmse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("MAE", f"{mae:.4f}")
    st.metric("R² Score", f"{r2:.4f}")

    # Load best model and prepare predictions
    best_model = load_model(".//models//stock_price.model.keras")

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    scaler = load_scaler()
    y_train_inv, y_train_pred_inv = inverse_transform_predictions(scaler, y_train, y_train_pred)
    y_val_inv, y_val_pred_inv = inverse_transform_predictions(scaler, y_val, y_val_pred)
    y_test_inv, y_test_pred_inv = inverse_transform_predictions(scaler, y_test, y_test_pred)

    # Plot actual vs predicted prices
    plot = plot_predictions(
        train_df["Date"][sequence_size:], val_df["Date"], test_df["Date"],
        y_train_inv, y_train_pred_inv,
        y_val_inv, y_val_pred_inv,
        y_test_inv, y_test_pred_inv, selected_ticker
    )
    st.pyplot(plot)

    # Predict future prices directly here
    with st.spinner("Predicting future prices..."):
        # Prepare last sequence for future predictions
        last_sequence = data_test_scaled[-sequence_size:]
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(prediction_days):
            current_sequence_reshaped = current_sequence.reshape(1, sequence_size, current_sequence.shape[1])
            next_pred = best_model.predict(current_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0])

            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred[0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row

        # Inverse transform future predictions
        future_predictions = np.array(future_predictions)
        dummy_array = np.zeros((len(future_predictions), data_test_scaled.shape[1]))
        dummy_array[:, 0] = future_predictions.reshape(-1)

        future_predictions_inv = scaler.inverse_transform(dummy_array)[:, 0]

        # Generate future dates
        last_date = test_df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

        # Plot future predictions
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(future_dates, future_predictions_inv, label='Predicted Future Prices', color='red', linestyle='-')
        ax.set_title(f'Future Price Predictions for {selected_ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.grid(True)
        ax.legend()
        locator = mdates.DayLocator(interval=2)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)