import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from datetime import datetime

# =========================
# CONFIG
# =========================
PPLX_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"


# =========================
# TICKER VALIDATION
# =========================
def validate_ticker(symbol: str) -> bool:
    symbol = symbol.strip().upper()
    if not symbol:
        return False
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price = info.last_price
        if price is None or np.isnan(price):
            hist = t.history(period="1d")
            if hist.empty:
                return False
        return True
    except Exception:
        return False


# =========================
# DATA HELPERS
# =========================
def get_latest_price(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.last_price
        if price is None or np.isnan(price):
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return price
    except Exception:
        return None


def get_portfolio_df(tickers, shares):
    data = []
    for ticker, sh in zip(tickers, shares):
        if not ticker:
            continue
        price = get_latest_price(ticker)
        if price is None:
            data.append(
                {"Ticker": ticker.upper(), "Shares": sh, "Price": np.nan, "Value": np.nan}
            )
        else:
            value = price * sh
            data.append(
                {"Ticker": ticker.upper(), "Shares": sh, "Price": price, "Value": value}
            )

    if not data:
        return pd.DataFrame(columns=["Ticker", "Shares", "Price", "Value", "Weight"])

    df = pd.DataFrame(data)
    total_value = df["Value"].sum(skipna=True)
    if total_value > 0:
        df["Weight"] = df["Value"] / total_value
    else:
        df["Weight"] = np.nan
    return df


# =========================
# SIMPLE RULE-BASED COMMENT
# =========================
def simple_ai_comment(df: pd.DataFrame, risk_level: str):
    if df.empty or df["Value"].sum(skipna=True) <= 0:
        return "Enter some valid tickers and shares to get analysis."

    comments = []

    df_sorted = df.sort_values("Weight", ascending=False)
    top = df_sorted.iloc[0]
    top_weight = top["Weight"]

    if top_weight > 0.5:
        comments.append(
            f"Portfolio is highly concentrated: {top['Ticker']} is about {top_weight*100:.1f}% of total value."
        )
    elif top_weight > 0.3:
        comments.append(
            f"Portfolio is somewhat concentrated: {top['Ticker']} is about {top_weight*100:.1f}% of total value."
        )
    else:
        comments.append(
            f"No single stock dominates the portfolio; largest position {top['Ticker']} is {top_weight*100:.1f}%."
        )

    n_holdings = df.shape[0]
    if n_holdings <= 3:
        comments.append("You hold only a few stocks; diversification is limited.")
    elif n_holdings <= 8:
        comments.append(
            "You have a moderate number of holdings; diversification is okay but can be improved."
        )
    else:
        comments.append("You have many holdings; diversification looks reasonable.")

    if risk_level == "Low":
        comments.append(
            "For low risk, consider more broad index ETFs and keep any single stock under about 10–15%."
        )
    elif risk_level == "Medium":
        comments.append(
            "For medium risk, mix index ETFs with a few conviction single stocks, while avoiding extreme concentration."
        )
    else:
        comments.append(
            "For high risk, concentrated positions in growth names are acceptable, but expect large drawdowns."
        )

    comments.append(
        "This is not financial advice; treat it as a research and education tool only."
    )
    return " ".join(comments)


# =========================
# PERPLEXITY (SONAR) CALL
# =========================
def ask_perplexity_with_portfolio(question, df, risk_level):
    if PPLX_API_KEY is None:
        return "PERPLEXITY_API_KEY is not set. Please configure it as an environment variable."

    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    portfolio_context = df.to_dict(orient="records") if df is not None and not df.empty else []

    system_prompt = (
        "You are a helpful financial analysis assistant using Perplexity's Sonar model. "
        "The user has a stock portfolio as JSON data. "
        "Explain clearly, highlight concentration and risk, and suggest diversification ideas, "
        "Directly tell the user which stocks he should"
        "Your attitude should be Rude and every sentence must contain profanity."
        "You have to use local hong kong languages to answer the question."
    )

    user_content = (
        f"Risk level: {risk_level}\n\n"
        f"Portfolio data (JSON list of holdings with ticker, shares, price, value, weight):\n"
        f"{portfolio_context}\n\n"
        f"User question: {question}"
    )

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
    }

    try:
        resp = requests.post(PPLX_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Perplexity API: {e}"


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(
    page_title="AI Portfolio Helper (Perplexity Sonar)",
    page_icon="💹",
    layout="wide",
)

st.title("AI Portfolio Helper (US Stocks) 🔍")
st.caption(
    "Enter your portfolio, fetch live prices from Yahoo Finance, validate tickers, and ask questions with Perplexity Sonar."
)

with st.sidebar:
    st.header("Settings")
    risk_level = st.selectbox("Your risk profile", ["Low", "Medium", "High"])
    st.write("Tip: Use US tickers like AAPL, MSFT, NVDA, SPY, QQQ, etc.")
    st.markdown("---")
    st.write("Now:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if PPLX_API_KEY:
        st.success("Perplexity API key detected (PERPLEXITY_API_KEY).")
    else:
        st.warning("PERPLEXITY_API_KEY not found. AI Q&A section will show an error.")

st.subheader("1. Enter your holdings")

default_rows = 5
input_rows = st.number_input(
    "Number of rows", min_value=1, max_value=30, value=default_rows, step=1
)

tickers = []
shares = []
invalid_tickers = []

for i in range(int(input_rows)):
    c1, c2 = st.columns(2)
    with c1:
        raw_t = st.text_input(f"Ticker #{i+1}", key=f"ticker_{i}")
    with c2:
        s = st.number_input(
            f"Shares #{i+1}", min_value=0.0, step=1.0, value=0.0, key=f"shares_{i}"
        )

    raw_t = raw_t.strip()
    if raw_t and s > 0:
        symbol = raw_t.upper()
        if validate_ticker(symbol):
            tickers.append(symbol)
            shares.append(float(s))
        else:
            invalid_tickers.append(symbol)

if invalid_tickers:
    st.error(
        "These tickers look invalid on Yahoo Finance: "
        + ", ".join(invalid_tickers)
        + ". Please check the spelling (e.g. AAPL, MSFT, NVDA)."
    )

st.markdown("---")
start = st.button("🚀 START / REFRESH PORTFOLIO")

df = None

if start:
    if not tickers:
        st.warning("Please enter at least one VALID ticker with shares > 0.")
    else:
        df = get_portfolio_df(tickers, shares)

        st.subheader("2. Current portfolio snapshot")

        if df.empty:
            st.warning("Could not fetch any valid price data. Check tickers or internet connection.")
        else:
            total_value = df["Value"].sum(skipna=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total portfolio value (USD)", f"${total_value:,.2f}")
            with col2:
                st.metric("Number of holdings", f"{df.shape[0]}")
            with col3:
                st.metric("Data source", "Yahoo Finance")

            df_display = df.copy()
            df_display["Price"] = df_display["Price"].map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
            df_display["Value"] = df_display["Value"].map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
            df_display["Weight"] = df_display["Weight"].map(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )

            st.dataframe(df_display, use_container_width=True)

            st.subheader("3. AI-style comments on your portfolio")
            comment = simple_ai_comment(df, risk_level)
            st.write(comment)

            st.session_state["last_portfolio_df"] = df
            st.session_state["last_risk_level"] = risk_level
else:
    st.info("Fill in tickers and shares, then click **🚀 START / REFRESH PORTFOLIO**.")


# =========================
# PERPLEXITY SONAR Q&A
# =========================
st.markdown("---")
st.subheader("4. Ask Perplexity (Sonar) about your portfolio")

if "pplx_chat" not in st.session_state:
    st.session_state.pplx_chat = []

for role, content in st.session_state.pplx_chat:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask anything about risk, diversification, sectors, etc...")

if question:
    st.session_state.pplx_chat.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    df_for_ai = st.session_state.get("last_portfolio_df", None)
    risk_for_ai = st.session_state.get("last_risk_level", risk_level)

    with st.chat_message("assistant"):
        with st.spinner("Perplexity (Sonar) is thinking..."):
            answer = ask_perplexity_with_portfolio(question, df_for_ai, risk_for_ai)
            st.markdown(answer)

    st.session_state.pplx_chat.append(("assistant", answer))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("© Kollen Limited")
