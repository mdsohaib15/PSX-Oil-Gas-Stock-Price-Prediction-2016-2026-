import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set up the page configuration
st.set_page_config(
    page_title="PSX Oil Sector Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sqlite3
import os

# Load and preprocess data
@st.cache_data
def load_data():
    db_file = 'data/psx_oil_data.db'
    
    # Use database if it exists (much faster and more professional)
    if os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM stock_data", conn)
        conn.close()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
        
    # Fallback to CSV if DB is not found
    df = pd.read_csv('data/PSX_Oil_Sector_Combined_2016_2026(Sheet1).csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Helper to convert Volume column (like 1.05K, 2.5M) to numeric
    def parse_volume(x):
        if pd.isna(x):
            return 0
        x = str(x).replace(',', '')
        if 'K' in x:
            return float(x.replace('K', '')) * 1e3
        elif 'M' in x:
            return float(x.replace('M', '')) * 1e6
        elif 'B' in x:
            return float(x.replace('B', '')) * 1e9
        else:
            try:
                return float(x)
            except ValueError:
                return 0
                
    df['Volume'] = df['Vol.'].apply(parse_volume)
    
    # Ensure numerical columns are correctly formatted
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
    return df

# Header
st.title("🛢️ PSX Oil & Gas Sector Dashboard (2016-2026)")

# Load data
try:
    with st.spinner("Loading data..."):
        df = load_data()
        companies = sorted(df['Symbol'].unique().tolist())
        
        # Get a mapped list of Company Name (Symbol)
        unique_comp_df = df[['Company', 'Symbol']].dropna().drop_duplicates().sort_values(by='Company')
        formatted_company_names = [f"{row['Company']} ({row['Symbol']})" for _, row in unique_comp_df.iterrows()]
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Create Tabs for sections
tab_overview, tab_data, tab_eda, tab_ml, tab_prediction, tab_manual = st.tabs([
    "Overview", 
    "Data",
    "EDA", 
    "ML Models", 
    "Future Values", 
    "Manual Prediction"
])

# ----------------- 1. Overview Section -----------------
with tab_overview:
    st.header("Overview")
    st.write("Welcome to the **PSX Oil & Gas Sector Data Analysis Dashboard**.")
    st.write("This application provides a comprehensive daily analysis of 17 different Oil & Gas companies listed on the Pakistan Stock Exchange, spanning a decade (2016-2026).")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Companies", len(companies))
    col2.metric("Total Trading Days", f"{len(df):,}")
    col3.metric("Date Range", f"{df['Date'].dt.date.min()} to {df['Date'].dt.date.max()}")
    
    st.markdown("### Companies Included")
    companies_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(formatted_company_names)])
    st.info(companies_list)

    st.markdown("### Company Scorecards (2016-2026)")
    df_sorted = df.sort_values(by='Date')
    
    # Create 6 columns for the scorecards to match the grid layout (17 total)
    cols = st.columns(6)
    
    for i, symbol in enumerate(companies):
        comp_data = df_sorted[df_sorted['Symbol'] == symbol].copy()
        if not comp_data.empty:
            first_price = comp_data.iloc[0]['Price']
            last_price = comp_data.iloc[-1]['Price']
            
            if pd.isna(first_price) or first_price == 0:
                total_return = 0
            else:
                total_return = ((last_price - first_price) / first_price) * 100
                
            avg_price = comp_data['Price'].mean()
            
            if 'Change %' in comp_data.columns:
                comp_data['Change_Num'] = pd.to_numeric(comp_data['Change %'].astype(str).str.replace('%', ''), errors='coerce')
                volatility = comp_data['Change_Num'].std() * np.sqrt(252) * 100
                if pd.isna(volatility):
                    volatility = (comp_data['Price'].std() / avg_price) * 100 if avg_price else 0
            else:
                volatility = (comp_data['Price'].std() / avg_price) * 100 if avg_price else 0
                
            with cols[i % 6]:
                # Streamlit metric gives us exactly the layout in the screenshot
                st.metric(
                    label="Total Return 2016-26",
                    value=symbol,
                    delta=f"{total_return:.1f}%"
                )
                st.caption(f"Avg PKR {avg_price:,.0f} - σ {volatility:.1f}%")

# ----------------- 2. Data Section -----------------
with tab_data:
    st.header("Data Overview")
    
    # Dropdown to filter data table by company
    selected_company_data = st.selectbox("Filter Dataset by Company", ["All Companies"] + companies, key='data_select')
    
    if selected_company_data == "All Companies":
        filtered_data_df = df
    else:
        filtered_data_df = df[df['Symbol'] == selected_company_data]
        
    st.write(f"Displaying raw data for: **{selected_company_data}** ({len(filtered_data_df):,} records)")
    
    # Display dataframe, dropping the intermediate column used in EDA
    display_df = filtered_data_df.drop(columns=['YearMonth'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("### Descriptive Statistics")
    st.dataframe(display_df.describe(), use_container_width=True)

# ----------------- 3. EDA Section -----------------
with tab_eda:
    st.header("Exploratory Data Analysis (EDA)")
    
    # Dropdown to filter by company
    selected_company_eda = st.selectbox("Select Company to View", ["All Companies"] + companies, key='eda_select')
    
    if selected_company_eda == "All Companies":
        filtered_df = df
        color_col = 'Symbol'
    else:
        filtered_df = df[df['Symbol'] == selected_company_eda]
        color_col = None
        
    st.markdown("---")
    
    # 1. Price Trend & Moving Averages / Candlestick
    if selected_company_eda != "All Companies":
        filtered_df = filtered_df.sort_values(by='Date')
        filtered_df['MA50'] = filtered_df['Price'].rolling(window=50).mean()
        filtered_df['MA200'] = filtered_df['Price'].rolling(window=200).mean()
        
        col_charts1, col_charts2 = st.columns(2)
        
        with col_charts1:
            st.subheader("Price Trend & Moving Averages")
            fig_ma = px.line(
                filtered_df, 
                x='Date', 
                y=['Price', 'MA50', 'MA200'], 
                title=f"50-Day & 200-Day Moving Averages - {selected_company_eda}",
                color_discrete_map={"Price": "white", "MA50": "cyan", "MA200": "orange"}
            )
            fig_ma.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_ma, use_container_width=True)
            
        with col_charts2:
            st.subheader("Candlestick Chart")
            fig_candle = go.Figure(data=[go.Candlestick(x=filtered_df['Date'],
                            open=filtered_df['Open'],
                            high=filtered_df['High'],
                            low=filtered_df['Low'],
                            close=filtered_df['Price'])])
            fig_candle.update_layout(template="plotly_dark", title=f"Candlestick Chart - {selected_company_eda}", margin=dict(l=20, r=20, t=40, b=20))
            # Hide the rangeslider for a cleaner look
            fig_candle.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True)
    else:
        st.subheader("Price Trend (Line Plot)")
        fig_line = px.line(
            filtered_df, 
            x='Date', 
            y='Price', 
            color=color_col,
            title="Closing Price History - All Companies"
        )
        fig_line.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)
        
    # 2. Daily Volatility (Returns)
    st.subheader("Daily Percentage Change (Volatility)")
    if 'Change %' in filtered_df.columns:
        filtered_df['Change_Num'] = pd.to_numeric(filtered_df['Change %'].astype(str).str.replace('%', ''), errors='coerce')
        fig_change = px.line(
            filtered_df, 
            x='Date', 
            y='Change_Num', 
            color=color_col,
            title=f"Daily Percentage Change - {selected_company_eda}"
        )
        fig_change.update_layout(template="plotly_dark", hovermode="x unified", yaxis_title="Daily Change (%)")
        st.plotly_chart(fig_change, use_container_width=True)

    # Create two columns for Histogram and Bar Plot
    col_hist, col_bar = st.columns(2)
    
    with col_hist:
        # 2. Histogram (Price Distribution)
        st.subheader("Price Distribution (Histogram)")
        fig_hist = px.histogram(
            filtered_df, 
            x='Price', 
            nbins=50, 
            color=color_col,
            title=f"Price Distribution - {selected_company_eda}",
            opacity=0.7
        )
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_bar:
        # 3. Bar Plot (Volume)
        st.subheader("Trading Volume (Bar Plot)")
        # Resampling data for cleaner bar charts
        filtered_df['YearMonth'] = filtered_df['Date'].dt.to_period('M').astype(str)
        vol_df = filtered_df.groupby(['YearMonth', 'Symbol'])['Volume'].sum().reset_index()
        
        fig_bar = px.bar(
            vol_df, 
            x='YearMonth', 
            y='Volume', 
            color=color_col,
            title=f"Total Monthly Trading Volume - {selected_company_eda}"
        )
        fig_bar.update_layout(template="plotly_dark", xaxis_title="Month", yaxis_title="Total Volume")
        fig_bar.update_xaxes(nticks=10) # Reduce x-axis labels clutter
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- 4. ML Models Section -----------------
with tab_ml:
    st.header("Machine Learning Models")
    
    try:
        import joblib
        # Load tuned Random Forest model + its feature scaler
        model  = joblib.load('models/tuned_rf.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        
        NUMERIC_FEATURES = [
            'Price', 'Open', 'High', 'Low', 'Volume',
            'Price_Yesterday', 'Price_5_Days_Ago',
            'MA7', 'MA30', 'Day_Of_Week'
        ]
        
        st.success("✅ **Tuned Random Forest Model successfully loaded and running!**")
        
    # Display Metrics
        st.subheader("🏆 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        # R-Squared with delta showing comparison to Train
        col1.metric("Model Accuracy", "99.84%", delta="+0.03%", delta_color="normal")

        # MAE is the average real-world miss
        col2.metric("Avg. Miss (MAE)", "2.21", help="Average difference between predicted and actual price")

        # MAPE as a percentage
        col3.metric("Error Rate (MAPE)", "2.13%")

        # RMSE or RSE for variance
        col4.metric("Volatility (RMSE)", "4.08")
        
        # Feature Importance Analysis
        st.markdown("---")
        st.subheader("🧠 Model Feature Analysis")
        if hasattr(model, 'feature_importances_'):
            st.write("This chart shows which features the Tuned Random Forest model relies on the most when predicting future stock prices.")
            importances = model.feature_importances_
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                features = [f"Feature {i}" for i in range(len(importances))]
            
            # Create a dataframe for plotting
            import pandas as pd
            import plotly.express as px
            feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat = px.bar(
                feat_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale="Viridis"
            )
            fig_feat.update_layout(template="plotly_dark", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature importances are not available for this model.")

        st.markdown("---")
        st.subheader("🔮 Live Price Predictor")
        st.write("Select a stock below. The AI will grab today's latest data, calculate the required features, and predict tomorrow's closing price!")
        
        # Dropdown for prediction
        pred_symbol = st.selectbox("Select Company Symbol:", df['Symbol'].unique(), key="pred_company")
        
        if st.button(f"Predict Next Day Price for {pred_symbol}", type="primary", use_container_width=True):
            with st.spinner(f"Running calculations for {pred_symbol}..."):
                # 1. Get latest data for this company
                comp_data = df[df['Symbol'] == pred_symbol].sort_values('Date').copy()
                
                # 2. Engineer the required features
                comp_data['Price_Yesterday'] = comp_data['Price'].shift(1).bfill()
                comp_data['Price_5_Days_Ago'] = comp_data['Price'].shift(5).bfill()
                comp_data['MA7'] = comp_data['Price'].rolling(window=7, min_periods=1).mean()
                comp_data['MA30'] = comp_data['Price'].rolling(window=30, min_periods=1).mean()
                comp_data['Day_Of_Week'] = comp_data['Date'].dt.dayofweek
                
                # 3. Extract the very last row
                latest_row = comp_data.iloc[-1:].copy()
                last_date = latest_row['Date'].values[0]
                last_price = latest_row['Price'].values[0]
                
                # 4. Build + scale the 10 numeric features
                X_pred = latest_row[NUMERIC_FEATURES].copy()
                X_pred[NUMERIC_FEATURES] = scaler.transform(X_pred[NUMERIC_FEATURES])
                
                # 5. One-Hot Encode Symbol
                for sym in sorted(df['Symbol'].unique()):
                    X_pred[f'Symbol_{sym}'] = 1 if sym == pred_symbol else 0
                
                # 6. Reorder columns to exactly match the trained model
                X_pred_final = X_pred[model.feature_names_in_]
                
                # 7. Predict
                prediction = model.predict(X_pred_final)[0]
                
                # 8. Display Results
                st.markdown("### 📊 Prediction Results")
                last_date_str = pd.to_datetime(last_date).strftime("%B %d, %Y")
                diff = prediction - last_price
                diff_pct = (diff / last_price) * 100
                
                r1, r2, r3 = st.columns(3)
                r1.metric("Latest Known Date", last_date_str)
                r2.metric("Latest Actual Price", f"Rs. {last_price:.2f}")
                r3.metric(
                    "🤖 Predicted Next Price",
                    f"Rs. {prediction:.2f}",
                    f"{diff:+.2f} ({diff_pct:+.2f}%)"
                )
                
    except FileNotFoundError:
        st.error("❌ Model file not found!")
        st.info("Make sure `models/tuned_rf.pkl` exists in your project directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ----------------- 5. Prediction Section -----------------
with tab_prediction:
    st.header("🔮 15-Day Stock Forecast (All Companies)")
    st.write("Autoregressive 15-day predictions powered by your AI model. Scroll right to see future dates!")
    
    try:
        import joblib
        model  = joblib.load('models/tuned_rf.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        
        NUMERIC_FEATURES = [
            'Price', 'Open', 'High', 'Low', 'Volume',
            'Price_Yesterday', 'Price_5_Days_Ago',
            'MA7', 'MA30', 'Day_Of_Week'
        ]
        
        # Function to generate 15 days of future data
        def get_15_day_forecast(company_symbol):
            comp_data = df[df['Symbol'] == company_symbol].sort_values('Date').copy()
            last_30_prices = comp_data['Price'].tail(30).tolist()
            last_volume = comp_data['Volume'].iloc[-1]
            current_date = pd.to_datetime(comp_data['Date'].iloc[-1])
            
            dates, predictions, changes = [], [], []
            
            for _ in range(15):
                # Skip weekends
                current_date += pd.Timedelta(days=1)
                if current_date.dayofweek == 5: current_date += pd.Timedelta(days=2)
                elif current_date.dayofweek == 6: current_date += pd.Timedelta(days=1)
                
                dates.append(current_date)
                
                # Extract moving averages and lags from our rolling history
                current_price = last_30_prices[-1]
                price_yesterday = last_30_prices[-2] if len(last_30_prices) >= 2 else current_price
                price_5_days_ago = last_30_prices[-6] if len(last_30_prices) >= 6 else current_price
                
                ma7 = np.mean(last_30_prices[-7:])
                ma30 = np.mean(last_30_prices[-30:])
                
                # Build + scale the 10 numeric features
                row_data = pd.DataFrame([{
                    'Price': current_price, 'Open': current_price,
                    'High': current_price, 'Low': current_price,
                    'Volume': last_volume,
                    'Price_Yesterday': price_yesterday,
                    'Price_5_Days_Ago': price_5_days_ago,
                    'MA7': ma7, 'MA30': ma30,
                    'Day_Of_Week': current_date.dayofweek
                }])
                row_data[NUMERIC_FEATURES] = scaler.transform(row_data[NUMERIC_FEATURES])
                
                # One-Hot Encode Symbol
                for sym in sorted(df['Symbol'].unique()):
                    row_data[f'Symbol_{sym}'] = 1 if sym == company_symbol else 0
                
                # Predict next day
                pred_price = model.predict(row_data[model.feature_names_in_])[0]
                pred_change = ((pred_price - current_price) / current_price) * 100
                
                predictions.append(pred_price)
                changes.append(pred_change)
                
                # Update rolling history with the new prediction
                last_30_prices.append(pred_price)
                last_30_prices.pop(0)
                
            return dates, predictions, changes

        import streamlit.components.v1 as components
        
        # Render a weather-like horizontal widget for all 17 companies
        for company in companies:
            dates, preds, changes = get_15_day_forecast(company)
            
            # HTML/CSS for Weather-style UI with Arrows
            widget_id = company.replace(' ', '_').replace('.', '')
            
            html_code = f"""
            <div style="font-family: sans-serif; margin-bottom: 10px;">
                <h3 style="color: white; margin-bottom: 5px; font-weight: 600;">🏢 {company}</h3>
                <div style="display: flex; align-items: center; background-color: #1e1e1e; border-radius: 10px; border: 1px solid #333;">
                    
                    <!-- Left Arrow -->
                    <button onclick="document.getElementById('scroll_{widget_id}').scrollBy(-200, 0)" style="background: none; border: none; color: white; font-size: 24px; cursor: pointer; padding: 15px; border-right: 1px solid #333;">&#10094;</button>
                    
                    <!-- Scrolling Container -->
                    <div id="scroll_{widget_id}" style="display: flex; overflow-x: auto; padding: 15px 0; scroll-behavior: smooth; width: 100%; -ms-overflow-style: none; scrollbar-width: none;">
                        <style>#scroll_{widget_id}::-webkit-scrollbar {{ display: none; }}</style>
            """
            
            for d, p, c in zip(dates, preds, changes):
                day_name = d.strftime("%a")
                day_num = d.strftime("%d %b")
                
                color = "#00fa9a" if c >= 0 else "#ff4b4b"
                icon = "📈" if c >= 0 else "📉"
                sign = "+" if c >= 0 else ""
                
                html_code += f"""
                        <!-- Individual Day Card -->
                        <div style="min-width: 110px; text-align: center; border-right: 1px solid #333; padding: 0 10px;">
                            <div style="font-size: 14px; color: #aaa; font-weight: bold;">{day_name}</div>
                            <div style="font-size: 14px; color: #ddd; margin-bottom: 5px;">{day_num}</div>
                            <div style="font-size: 30px; margin: 10px 0;">{icon}</div>
                            <div style="font-size: 18px; color: #fff; font-weight: bold;">Rs {p:.1f}</div>
                            <div style="font-size: 15px; color: {color}; font-weight: bold; margin-top: 5px;">{sign}{c:.2f}%</div>
                        </div>
                """
                
            html_code += f"""
                    </div>
                    
                    <!-- Right Arrow -->
                    <button onclick="document.getElementById('scroll_{widget_id}').scrollBy(200, 0)" style="background: none; border: none; color: white; font-size: 24px; cursor: pointer; padding: 15px; border-left: 1px solid #333;">&#10095;</button>
                </div>
            </div>
            """
            
            # Embed the HTML inside Streamlit
            components.html(html_code, height=240)
            
    except Exception as e:
        st.error(f"Error generating 15-day predictions: {e}")

# ----------------- 6. Manual Prediction Section -----------------
with tab_manual:
    st.header("🔮 Manual Price Prediction")
    st.write("Enter technical values manually to get a next-day price prediction from the AI model.")
    
    try:
        import joblib
        # Load tuned Random Forest model + its feature scaler
        model  = joblib.load('models/tuned_rf.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        
        NUMERIC_FEATURES = [
            'Price', 'Open', 'High', 'Low', 'Volume',
            'Price_Yesterday', 'Price_5_Days_Ago',
            'MA7', 'MA30', 'Day_Of_Week'
        ]
        
        all_symbols = sorted(df['Symbol'].unique().tolist())
        
        with st.form("manual_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Core Market Data")
                symbol = st.selectbox("Company Symbol", all_symbols)
                price = st.number_input("Today's Closing Price (PKR)", value=100.0, step=0.1)
                open_p = st.number_input("Today's Open Price", value=100.0, step=0.1)
                high = st.number_input("Today's High Price", value=102.0, step=0.1)
                low = st.number_input("Today's Low Price", value=98.0, step=0.1)
                volume = st.number_input("Today's Volume", value=1000000, step=1000)
            
            with col2:
                st.subheader("Historical & Time Data")
                price_yesterday = st.number_input("Yesterday's Price", value=99.0, step=0.1)
                price_5_days = st.number_input("Price 5 Days Ago", value=95.0, step=0.1)
                ma7 = st.number_input("7-Day Moving Average (MA7)", value=98.0, step=0.1)
                ma30 = st.number_input("30-Day Moving Average (MA30)", value=90.0, step=0.1)
                day_of_week = st.selectbox("Day of Week", options=list(range(7)), 
                                         format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
            
            submit = st.form_submit_button("Predict Tomorrow's Price", type="primary", use_container_width=True)
            
        if submit:
            # Build feature row
            input_df = pd.DataFrame([{
                'Price': price, 'Open': open_p, 'High': high, 'Low': low,
                'Volume': volume, 'Price_Yesterday': price_yesterday,
                'Price_5_Days_Ago': price_5_days,
                'MA7': ma7, 'MA30': ma30,
                'Day_Of_Week': day_of_week
            }])
            
            # Scale the 10 numeric features
            input_df[NUMERIC_FEATURES] = scaler.transform(input_df[NUMERIC_FEATURES])
            
            # One-Hot Encode Symbol
            for sym in all_symbols:
                input_df[f'Symbol_{sym}'] = 1 if sym == symbol else 0
            
            # Predict using model's exact feature order
            prediction = model.predict(input_df[model.feature_names_in_])[0]
            
            # Display Results
            st.markdown("---")
            st.success(f"### 🎯 Predicted Price: Rs. {prediction:.2f}")
            
            r1, r2 = st.columns(2)
            diff = prediction - price
            diff_pct = (diff / price) * 100
            r1.metric("Predicted Price Change", f"Rs. {diff:+.2f}", f"{diff_pct:+.2f}%")
            r2.info(f"Model used: Random Forest")

    except Exception as e:
        st.error(f"Error in Manual Prediction: {e}")

