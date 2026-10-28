import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Forecaster", page_icon="📈", layout="wide")

# --- API URL DETECTION (Streamlit Secrets vs Localhost) ---
try:
    # When deployed on Streamlit Cloud, it will read the API_URL from the Secrets tab
    API_URL = st.secrets["API_URL"]
except Exception:
    # When running locally on your laptop
    API_URL = "http://127.0.0.1:8000"
# -----------------------------------------------------------

st.title("📈 Dynamic Stock Forecaster")
st.markdown("Search for any Indian or US stock to see the historical chart, technical indicators, and predictions.")

@st.cache_data(show_spinner="Loading stock database... (This only happens once)")
def get_stock_list():
    try:
        url = "https://huggingface.co/datasets/tickertruth/nse-india-security-master/resolve/main/data/nse_security_master.csv"
        df = pd.read_csv(url)
        
        if 'active_flag' in df.columns:
            df = df[df['active_flag'] == True]
            
        tickers_list = []
        
        for _, row in df.iterrows():
            symbol = str(row['nse_symbol']).strip()
            company = str(row['company_name']).strip()
            
            if symbol and symbol != 'nan':
                full_ticker = f"{symbol}.NS"
                display_label = f"{full_ticker} - {company}"
                tickers_list.append(display_label)
                
        us_stocks = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp."}
        for symbol, name in us_stocks.items():
            display_label = f"{symbol} - {name}"
            tickers_list.append(display_label)
            
        tickers_list = sorted(tickers_list)
        return tickers_list
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return ["AAPL - Apple Inc.", "MSFT - Microsoft Corp.", "TSLA - Tesla Inc."]

stock_list = get_stock_list()

# --- Search Bar and Analyze Button ---
col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")

with col_a:
    selected_display = st.selectbox(
        "Search for a Stock (Type company name or ticker)", 
        options=stock_list,
        index=None,
        placeholder="Type 'vedant', 'adani', 'tata', or 'AAPL'...",
    )
with col_b:
    analyze_button = st.button("Analyze Stock", type="primary", use_container_width=True)

# --- Main Logic ---
if analyze_button and selected_display:
    ticker_search = selected_display.split(" - ")[0].strip()
    
    # Pre-flight check: Get total data days for backtest logic
    with st.spinner("Fetching data and checking history..."):
        try:
            init_response = requests.post(f"{API_URL}/predict", json={"ticker": ticker_search, "horizon": 5})
            if init_response.status_code == 200:
                data_days = init_response.json().get('data_days', 0)
            else:
                st.error(f"Failed to fetch data: {init_response.json().get('detail')}")
                data_days = 0
        except Exception as e:
            st.error(f"Failed to connect to API. Is FastAPI running? Error: {e}")
            data_days = 0

    if data_days > 0:
        tab_swing, tab_long, tab_invest, tab_bt_6m, tab_bt_1y = st.tabs([
            "📊 Swing Trade (1 Week)", 
            "📈 Long-Term (1 Month)", 
            "💼 Investing (1 Year)",
            "🏦 6-Month Backtest",
            "🏦 1-Year Backtest"
        ])
        
        def render_prediction(horizon_days, tab_obj):
            with tab_obj:
                with st.spinner(f"Calculating {horizon_days}-day prediction..."):
                    try:
                        payload = {"ticker": ticker_search, "horizon": horizon_days}
                        response = requests.post(f"{API_URL}/predict", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            currency = "₹" if ".NS" in ticker_search else "$"
                            
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                if result['direction'] == "UP":
                                    st.metric(label="Prediction", value="UP 🟢", delta=result['expected_return_pct'])
                                else:
                                    st.metric(label="Prediction", value="DOWN 🔴", delta=result['expected_return_pct'], delta_color="inverse")
                            with c2:
                                st.metric(label="Current Price", value=f"{currency}{result['latest_close']}")
                            with c3:
                                st.metric(label=f"Target Price ({horizon_days}D)", value=f"{currency}{result['target_price']}")
                                
                            st.progress(result['confidence'], text=f"Model Confidence: {result['confidence'] * 100:.1f}%")
                            st.divider()
                            
                            df = pd.DataFrame(result['historical_data'])
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                            
                            fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df['date'], y=df['sma_10'], line=dict(color='orange', width=1.5), name="SMA 10"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df['date'], y=df['sma_50'], line=dict(color='blue', width=1.5), name="SMA 50"), row=1, col=1)
                            
                            if horizon_days > 1:
                                last_hist_date = str(df['date'].iloc[-1])
                                last_hist_price = float(df['close'].iloc[-1])
                                
                                f_dates = [last_hist_date] + list(result['future_dates'])
                                f_prices = [last_hist_price] + list(result['future_prices'])
                                
                                color = 'green' if result['direction'] == "UP" else 'red'
                                
                                fig.add_trace(go.Scatter(
                                    x=f_dates, y=f_prices, mode='lines',
                                    line=dict(color=color, width=2, dash='dash'),
                                    name=f"Projected {horizon_days}D"
                                ), row=1, col=1)
                            
                            fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color='gray'), row=2, col=1)
                            
                            fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{horizon_days}")
                        else:
                            st.error(f"Error: {response.json()['detail']}")
                    except Exception as e:
                        st.error(f"Failed to connect to API. Error: {e}")

        # Render the 3 prediction tabs
        render_prediction(5, tab_swing)
        render_prediction(21, tab_long)
        render_prediction(252, tab_invest)

        # --- Backtest Logic Helper ---
        def render_backtest(days, tab_obj, title_text):
            with tab_obj:
                if data_days < days:
                    st.warning(f"⚠️ **Cannot run {title_text} Backtest.**")
                    st.info(f"This stock only has **{data_days}** trading days of history. "
                            f"It needs at least **{days}** days. This usually happens with newly listed or spun-off companies.")
                else:
                    with st.spinner(f"Running {title_text} Backtest..."):
                        try:
                            bt_response = requests.get(f"{API_URL}/backtest/{ticker_search}?days={days}")
                            
                            if bt_response.status_code == 200:
                                bt_data = bt_response.json()
                                currency = "₹" if ".NS" in ticker_search else "$"
                                
                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    st.metric(label="Initial Capital", value=f"{currency}10,000")
                                with m2:
                                    delta_color = "normal" if bt_data['strategy_return_pct'] >= bt_data['buy_hold_return_pct'] else "inverse"
                                    st.metric(
                                        label="AI Strategy Return", 
                                        value=f"{bt_data['strategy_return_pct']}%", 
                                        delta=f"{bt_data['strategy_return_pct'] - bt_data['buy_hold_return_pct']:.2f}% vs Buy&Hold", 
                                        delta_color=delta_color
                                    )
                                with m3:
                                    st.metric(label="Buy & Hold Return", value=f"{bt_data['buy_hold_return_pct']}%")
                                
                                st.divider()
                                
                                fig_bt = go.Figure()
                                fig_bt.add_trace(go.Scatter(x=bt_data['dates'], y=bt_data['strategy_equity'], mode='lines', name='AI Strategy', line=dict(color='green', width=2)))
                                fig_bt.add_trace(go.Scatter(x=bt_data['dates'], y=bt_data['buy_hold_equity'], mode='lines', name='Buy & Hold', line=dict(color='gray', width=2, dash='dot')))
                                fig_bt.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Portfolio Value", height=500, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                                
                                st.plotly_chart(fig_bt, use_container_width=True, key=f"bt_chart_{days}")
                                st.info("**How this works:** The AI Strategy only holds the stock when the model predicts an UP day. If it predicts DOWN, it holds cash (0% return that day). This prevents large losses during downturns.")
                            else:
                                st.error(f"Backtest Error: {bt_response.json()['detail']}")
                        except Exception as e:
                            st.error(f"Failed to connect to API for backtest: {e}")

        # Render the backtest tabs
        render_backtest(126, tab_bt_6m, "6-Month")
        render_backtest(252, tab_bt_1y, "1-Year")

else:
    st.info("👆 Please search for a stock (e.g., 'vedant', 'adani', or 'AAPL') and click Analyze.")

st.divider()
st.caption("Built by Kiran Halepujari | XGBoost + FastAPI + Streamlit")