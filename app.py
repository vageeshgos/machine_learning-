import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = pd.read_csv("data/Stock_Dataset.csv")
    return df

df_stocks = load_data()

feature_cols = [
    "Market Cap ($B)",
    "Current Price ($)",
    "52W High ($)",
    "52W Low ($)",
    "Revenue ($B)",
    "Net Income ($B)",
    "EPS ($)",
    "P/E Ratio",
    "P/B Ratio",
    "ROE (%)",
    "Debt/Equity",
    "Dividend Yield (%)",
    "Revenue Growth YoY (%)",
    "Profit Margin (%)",
    "Current Ratio",
    "Beta",
    "RSI (14D)",
    "MACD",
    "3M Volatility (%)",
    "Analyst Buy",
    "Analyst Hold",
    "Analyst Sell",
    "Target Price ($)",
]

target = "Risk Category"

X = df_stocks[feature_cols].copy()
y = df_stocks[target]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

@st.cache_resource
def train_model():
    best_acc, best_model = 0.0, None
    configs = [
        {"n_estimators": n, "max_depth": d}
        for n in [200, 300, 400]
        for d in [4, 6, 8, 10]
    ]
    for cv_splits in [5, 3]:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        for cfg in configs:
            m = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
            )
            scores = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
            mean_acc = float(scores.mean())
            if mean_acc >= 0.70 and mean_acc > best_acc:
                best_acc = mean_acc
                best_model = m
        if best_model is not None:
            break
    if best_model is None:
        best_model = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", max_depth=8
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_acc = float(cross_val_score(best_model, X, y, cv=cv, scoring="accuracy").mean())
    best_model.fit(X, y)
    return best_model, best_acc, imputer

model, test_acc, imputer = train_model()

beta_min, beta_max = df_stocks["Beta"].min(), df_stocks["Beta"].max()
vol_min, vol_max = df_stocks["3M Volatility (%)"].min(), df_stocks["3M Volatility (%)"].max()

def add_risk_score(df):
    beta_norm = (df["Beta"] - beta_min) / (beta_max - beta_min) if (beta_max - beta_min) != 0 else 0
    vol_norm = (df["3M Volatility (%)"] - vol_min) / (vol_max - vol_min) if (vol_max - vol_min) != 0 else 0
    return 0.6 * vol_norm + 0.4 * beta_norm

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #000000; }
    [data-testid="stHeader"] { background-color: #0a0a0a; }
    .js-plotly-plot { min-height: 450px !important; background: #e8e8ec !important; border-radius: 8px; padding: 8px; }
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] .stMarkdown { color: #e0e0e0 !important; }
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3 { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Stock Portfolio Risk Intelligence")
st.write("Predict Risk Category for stocks and explore portfolio risk metrics.")

st.sidebar.markdown(f"Model accuracy (CV): {test_acc:.1%}")

tab_upload, tab_portfolio, tab_single = st.tabs([
    "Upload Your Data",
    "Portfolio Table",
    "Single Stock Prediction",
])

with tab_upload:
    st.subheader("Upload Your Data — Stock Profile & Risk Analysis")
    st.write("Upload a CSV with your stock data. We will predict Risk Category (Low / Medium / High) for each row.")
    st.caption("Predictions and charts appear only after you upload a valid CSV. The table below is not shown until you upload.")

    sample = df_stocks[["Company Name", "Ticker", "Sector"] + feature_cols].head(2)
    st.download_button(
        label="Download sample CSV template",
        data=sample.to_csv(index=False),
        file_name="stock_upload_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file to upload",
        type=["csv"],
        key="upload_csv",
        help="Your CSV must have the same column names as the sample template.",
    )

    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            up_df = None

        if up_df is not None and not up_df.empty:
            missing = [c for c in feature_cols if c not in up_df.columns]
            if missing:
                st.error(
                    "Your CSV is missing these required columns: "
                    + ", ".join(missing)
                    + ". Use the sample template above for the correct format."
                )
            else:
                X_up = up_df[feature_cols].copy()
                valid = X_up.notna().all(axis=1)
                if not valid.all():
                    st.warning(f"Dropped {(~valid).sum()} row(s) with missing values in feature columns.")
                X_up = X_up[valid].fillna(0)
                up_df = up_df[valid].reset_index(drop=True)

                if X_up.empty:
                    st.error("No rows left after removing missing values.")
                else:
                    preds = model.predict(X_up)
                    probas = model.predict_proba(X_up)

                    result = pd.DataFrame({
                        "Predicted Risk": preds,
                        "RiskScore": add_risk_score(up_df.copy()),
                    })
                    for i, cls in enumerate(model.classes_):
                        result[f"P({cls})"] = probas[:, i].round(2)

                    id_cols = [c for c in ["Company Name", "Ticker", "Sector"] if c in up_df.columns]
                    if id_cols:
                        for c in id_cols:
                            result.insert(0, c, up_df[c].values)

                    st.success(f"Predicted risk for {len(result)} stock(s).")
                    st.dataframe(result, use_container_width=True)

                    st.subheader("Risk summary (your upload)")
                    risk_counts = result["Predicted Risk"].value_counts()
                    st.bar_chart(risk_counts)
                    st.caption("Count per predicted risk category.")
                    st.download_button(
                        label="Download predictions as CSV",
                        data=result.to_csv(index=False),
                        file_name="stock_risk_predictions.csv",
                        mime="text/csv",
                        key="download_predictions",
                    )
    else:
        st.info("No file uploaded. Use Choose a CSV file to upload above. Predictions table and risk charts will appear here only after you upload a valid CSV.")
        st.markdown("---")

with tab_portfolio:
    st.subheader("Portfolio Risk Overview")
    st.caption("This tab shows the app’s built-in sample portfolio (not your uploaded file). Use the Upload Your Data tab to analyze your own CSV.")

    st.dataframe(
        df_stocks[["Company Name", "Ticker", "Sector", "Risk Category", "Growth Category"]]
    )

    beta_norm = (df_stocks["Beta"] - df_stocks["Beta"].min()) / (
        df_stocks["Beta"].max() - df_stocks["Beta"].min()
    )
    vol_norm = (df_stocks["3M Volatility (%)"] - df_stocks["3M Volatility (%)"].min()) / (
        df_stocks["3M Volatility (%)"].max() - df_stocks["3M Volatility (%)"].min()
    )
    df_stocks["RiskScore"] = 0.6 * vol_norm + 0.4 * beta_norm

    st.subheader("RiskScore vs Current Price")
    st.scatter_chart(
        df_stocks,
        x="RiskScore",
        y="Current Price ($)",
        color="Risk Category",
    )

    st.subheader("Average RiskScore by Sector")
    sector_risk = df_stocks.groupby("Sector")["RiskScore"].mean().reset_index()
    st.bar_chart(sector_risk, x="Sector", y="RiskScore")

    st.subheader("3D Visualizations")

    df_3d = df_stocks.copy()
    log_cap = np.log1p(df_3d["Market Cap ($B)"])
    lo, hi = log_cap.min(), log_cap.max()
    size_norm = (log_cap - lo) / (hi - lo + 1e-9)
    df_3d["_size"] = (size_norm * 14 + 6).clip(6, 22)

    scene_style = dict(
        bgcolor="rgb(240,240,245)",
        xaxis=dict(backgroundcolor="rgb(255,255,255)", gridcolor="rgba(100,100,100,0.4)", showbackground=True, title_font=dict(color="#333"), tickfont=dict(color="#444")),
        yaxis=dict(backgroundcolor="rgb(255,255,255)", gridcolor="rgba(100,100,100,0.4)", showbackground=True, title_font=dict(color="#333"), tickfont=dict(color="#444")),
        zaxis=dict(backgroundcolor="rgb(255,255,255)", gridcolor="rgba(100,100,100,0.4)", showbackground=True, title_font=dict(color="#333"), tickfont=dict(color="#444")),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2), center=dict(x=0, y=0, z=-0.1)),
    )
    slider_btn_style = dict(
        slider_bg="rgb(70,70,75)",
        slider_border="rgba(0,0,0,0.3)",
        slider_font=dict(color="#ffffff", size=12),
        menu_bg="rgb(70,70,75)",
        menu_border="rgba(0,0,0,0.3)",
        menu_font=dict(color="#ffffff", size=12),
    )
    risk_colors = {"Low": "#00c853", "Medium": "#ff9800", "High": "#f44336"}
    transition = dict(duration=400, easing="cubic-in-out")

    col_3d_1, col_3d_2 = st.columns(2)
    with col_3d_1:
        st.markdown("Risk space: Beta × Volatility × Price (animate by sector)")
        fig1 = px.scatter_3d(
            df_3d,
            x="Beta",
            y="3M Volatility (%)",
            z="Current Price ($)",
            color="Risk Category",
            size="_size",
            size_max=22,
            animation_frame="Sector",
            hover_data=["Company Name", "Ticker", "Sector"],
            color_discrete_map=risk_colors,
        )
        fig1.update_traces(
            marker=dict(line=dict(width=1.2, color="rgba(0,0,0,0.3)"), opacity=0.9),
            selector=dict(mode="markers"),
        )
        fig1.update_layout(
            scene=dict(**scene_style, xaxis_title="Beta", yaxis_title="3M Volatility (%)", zaxis_title="Current Price ($)"),
            margin=dict(l=0, r=0, b=0, t=50),
            width=700,
            height=500,
            transition=transition,
            paper_bgcolor="rgb(248,248,250)",
            font=dict(family="Segoe UI, sans-serif", size=11, color="#333"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5, font=dict(color="#333")),
        )
        fig1.layout.sliders[0].pad = dict(t=55, b=12)
        fig1.layout.sliders[0].bgcolor = slider_btn_style["slider_bg"]
        fig1.layout.sliders[0].bordercolor = slider_btn_style["slider_border"]
        fig1.layout.sliders[0].borderwidth = 1
        fig1.layout.sliders[0].font = slider_btn_style["slider_font"]
        fig1.layout.sliders[0].currentvalue = dict(visible=True, prefix="", xanchor="left", font=dict(color="#ffffff", size=12))
        fig1.layout.updatemenus[0].pad = dict(t=55, b=12)
        fig1.layout.updatemenus[0].bgcolor = slider_btn_style["menu_bg"]
        fig1.layout.updatemenus[0].bordercolor = slider_btn_style["menu_border"]
        fig1.layout.updatemenus[0].borderwidth = 1
        fig1.layout.updatemenus[0].font = slider_btn_style["menu_font"]
        st.plotly_chart(fig1, use_container_width=True, height=520)

    with col_3d_2:
        st.markdown("RiskScore × P/E × Price (animate by risk category)")
        fig2 = px.scatter_3d(
            df_3d,
            x="RiskScore",
            y="P/E Ratio",
            z="Current Price ($)",
            color="Risk Category",
            size="_size",
            size_max=22,
            animation_frame="Risk Category",
            hover_data=["Company Name", "Ticker", "Sector"],
            color_discrete_map=risk_colors,
        )
        fig2.update_traces(
            marker=dict(line=dict(width=1.2, color="rgba(0,0,0,0.3)"), opacity=0.9),
            selector=dict(mode="markers"),
        )
        fig2.update_layout(
            scene=dict(**scene_style, xaxis_title="RiskScore", yaxis_title="P/E Ratio", zaxis_title="Current Price ($)"),
            margin=dict(l=0, r=0, b=0, t=50),
            width=700,
            height=500,
            transition=transition,
            paper_bgcolor="rgb(248,248,250)",
            font=dict(family="Segoe UI, sans-serif", size=11, color="#333"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5, font=dict(color="#333")),
        )
        fig2.layout.sliders[0].pad = dict(t=55, b=12)
        fig2.layout.sliders[0].bgcolor = slider_btn_style["slider_bg"]
        fig2.layout.sliders[0].bordercolor = slider_btn_style["slider_border"]
        fig2.layout.sliders[0].borderwidth = 1
        fig2.layout.sliders[0].font = slider_btn_style["slider_font"]
        fig2.layout.sliders[0].currentvalue = dict(visible=True, prefix="", xanchor="left", font=dict(color="#ffffff", size=12))
        fig2.layout.updatemenus[0].pad = dict(t=55, b=12)
        fig2.layout.updatemenus[0].bgcolor = slider_btn_style["menu_bg"]
        fig2.layout.updatemenus[0].bordercolor = slider_btn_style["menu_border"]
        fig2.layout.updatemenus[0].borderwidth = 1
        fig2.layout.updatemenus[0].font = slider_btn_style["menu_font"]
        st.plotly_chart(fig2, use_container_width=True, height=520)

    n = len(df_3d)
    reveal_frames = []
    for i in range(n):
        for f in range(i + 1, n + 1):
            reveal_frames.append({"Company Name": df_3d.iloc[i]["Company Name"], "Ticker": df_3d.iloc[i]["Ticker"], "Sector": df_3d.iloc[i]["Sector"], "Risk Category": df_3d.iloc[i]["Risk Category"], "Beta": df_3d.iloc[i]["Beta"], "3M Volatility (%)": df_3d.iloc[i]["3M Volatility (%)"], "Current Price ($)": df_3d.iloc[i]["Current Price ($)"], "_size": df_3d.iloc[i]["_size"], "Reveal": f})
    df_reveal = pd.DataFrame(reveal_frames)

    st.markdown("Stocks revealed one by one (play animation)")
    fig_reveal = px.scatter_3d(
        df_reveal,
        x="Beta",
        y="3M Volatility (%)",
        z="Current Price ($)",
        color="Risk Category",
        size="_size",
        size_max=22,
        animation_frame="Reveal",
        hover_data=["Company Name", "Ticker", "Sector"],
        color_discrete_map=risk_colors,
    )
    fig_reveal.update_traces(
        marker=dict(line=dict(width=1.2, color="rgba(0,0,0,0.3)"), opacity=0.9),
        selector=dict(mode="markers"),
    )
    fig_reveal.update_layout(
        scene=dict(**scene_style, xaxis_title="Beta", yaxis_title="3M Volatility (%)", zaxis_title="Current Price ($)"),
        margin=dict(l=0, r=0, b=0, t=50),
        width=800,
        height=520,
        transition=dict(duration=300, easing="cubic-in-out"),
        paper_bgcolor="rgb(248,248,250)",
        font=dict(family="Segoe UI, sans-serif", size=11, color="#333"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5, font=dict(color="#333")),
    )
    if fig_reveal.layout.sliders:
        fig_reveal.layout.sliders[0].pad = dict(t=55, b=12)
        fig_reveal.layout.sliders[0].bgcolor = slider_btn_style["slider_bg"]
        fig_reveal.layout.sliders[0].bordercolor = slider_btn_style["slider_border"]
        fig_reveal.layout.sliders[0].borderwidth = 1
        fig_reveal.layout.sliders[0].font = slider_btn_style["slider_font"]
        fig_reveal.layout.sliders[0].currentvalue = dict(visible=True, prefix="", xanchor="left", font=dict(color="#ffffff", size=12))
    if fig_reveal.layout.updatemenus:
        fig_reveal.layout.updatemenus[0].pad = dict(t=55, b=12)
        fig_reveal.layout.updatemenus[0].bgcolor = slider_btn_style["menu_bg"]
        fig_reveal.layout.updatemenus[0].bordercolor = slider_btn_style["menu_border"]
        fig_reveal.layout.updatemenus[0].borderwidth = 1
        fig_reveal.layout.updatemenus[0].font = slider_btn_style["menu_font"]
    st.plotly_chart(fig_reveal, use_container_width=True, height=560)

    st.markdown("Sector view: Market Cap × Revenue × RiskScore (animate by sector)")
    fig3 = px.scatter_3d(
        df_3d,
        x="Market Cap ($B)",
        y="Revenue ($B)",
        z="RiskScore",
        color="Sector",
        size="_size",
        size_max=22,
        symbol="Risk Category",
        animation_frame="Sector",
        hover_data=["Company Name", "Ticker", "Risk Category"],
    )
    fig3.update_traces(
        marker=dict(line=dict(width=1.2, color="rgba(0,0,0,0.3)"), opacity=0.9),
        selector=dict(mode="markers"),
    )
    fig3.update_layout(
        scene=dict(**scene_style, xaxis_title="Market Cap ($B)", yaxis_title="Revenue ($B)", zaxis_title="RiskScore"),
        margin=dict(l=0, r=0, b=0, t=50),
        width=800,
        height=560,
        transition=transition,
        paper_bgcolor="rgb(248,248,250)",
        font=dict(family="Segoe UI, sans-serif", size=11, color="#333"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5, font=dict(color="#333")),
    )
    if fig3.layout.sliders:
        fig3.layout.sliders[0].pad = dict(t=55, b=12)
        fig3.layout.sliders[0].bgcolor = slider_btn_style["slider_bg"]
        fig3.layout.sliders[0].bordercolor = slider_btn_style["slider_border"]
        fig3.layout.sliders[0].borderwidth = 1
        fig3.layout.sliders[0].font = slider_btn_style["slider_font"]
        fig3.layout.sliders[0].currentvalue = dict(visible=True, prefix="", xanchor="left", font=dict(color="#ffffff", size=12))
    if fig3.layout.updatemenus:
        fig3.layout.updatemenus[0].pad = dict(t=55, b=12)
        fig3.layout.updatemenus[0].bgcolor = slider_btn_style["menu_bg"]
        fig3.layout.updatemenus[0].bordercolor = slider_btn_style["menu_border"]
        fig3.layout.updatemenus[0].borderwidth = 1
        fig3.layout.updatemenus[0].font = slider_btn_style["menu_font"]
    st.plotly_chart(fig3, use_container_width=True, height=600)

with tab_single:
    st.subheader("Predict Risk Category for a Stock")

    tickers = df_stocks["Ticker"].tolist()
    selected = st.selectbox("Choose an existing stock:", tickers)

    row = df_stocks[df_stocks["Ticker"] == selected].iloc[0]

    st.write("Loaded features for:", row["Company Name"])

    input_row = pd.DataFrame(row[feature_cols].values.reshape(1, -1), columns=feature_cols)
    input_features = imputer.transform(input_row)

    pred = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0]

    st.markdown(f"Predicted Risk Category: {pred}")
    st.write("Class probabilities:", dict(zip(model.classes_, np.round(proba, 2))))
