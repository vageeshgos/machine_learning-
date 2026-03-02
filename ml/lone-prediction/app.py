import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import time, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="LoanGuard AI", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Sora:wght@300;400;600;800&display=swap');
*, html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp, .main { background-color: #060910; }

.hero {
    background: linear-gradient(135deg, #0d1f35, #060910);
    border: 1px solid #00d4aa40; border-radius: 20px;
    padding: 2.5rem; text-align: center; margin-bottom: 2rem;
}
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #0099ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #6e7a8a; font-size: 1rem; margin-top: 0.5rem; }

.upload-box {
    background: linear-gradient(135deg, #0d1f35, #060910);
    border: 2.5px dashed #00d4aa80; border-radius: 20px;
    padding: 3.5rem 2rem; text-align: center; margin: 1.5rem 0;
}
.upload-icon  { font-size: 4.5rem; }
.upload-title { color: #00d4aa; font-size: 1.8rem; font-weight: 700; margin: 0.8rem 0 0.4rem; }
.upload-sub   { color: #6e7a8a; font-size: 0.95rem; }

.sec-head {
    color: #00d4aa; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin: 2rem 0 0.8rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid #00d4aa25;
}

/* BIG RESULT BOX */
.result-wrap { border-radius: 22px; padding: 2.5rem; text-align: center; margin: 1.5rem 0; }
.result-high { background:#1a0000; border: 3px solid #ff3333; }
.result-mid  { background:#1a1000; border: 3px solid #ffaa00; }
.result-low  { background:#001a0d; border: 3px solid #00d4aa; }
.r-emoji  { font-size: 4.5rem; }
.r-label  { font-size: 1.5rem; font-weight: 700; letter-spacing: 0.06em; margin: 0.5rem 0; }
.r-pct    { font-size: 5.5rem; font-weight: 800; line-height: 1.1; }
.r-sub    { color: #8892a4; font-size: 0.95rem; margin: 0.4rem 0 1rem; }
.r-advice { display:inline-block; padding: 0.6rem 1.5rem; border-radius: 50px;
            font-size: 1rem; font-weight: 600; }
.bar-wrap  { background:#1a1f2e; border-radius:50px; height:20px; margin:1rem auto; max-width:400px; overflow:hidden; }
.bar-fill  { height:20px; border-radius:50px; }

.kcard { background:#0d1a2e; border-left:3px solid #00d4aa; border-radius:10px;
         padding:0.9rem 1.1rem; margin:0.4rem 0; color:#c9d1d9; font-size:0.86rem; }
.lcard { background:#150f00; border-left:3px solid #ffaa00; border-radius:10px;
         padding:0.9rem 1.1rem; margin:0.4rem 0; color:#c9a227; font-size:0.82rem; }
.tip   { background:#0d1117; border:1px dashed #00d4aa55; border-radius:10px;
         padding:0.8rem 1rem; color:#6e7a8a; font-size:0.85rem; margin:0.4rem 0; }
.g-title { color:#c9d1d9; font-size:1rem; font-weight:600; margin:1.2rem 0 0.3rem; }

.stButton > button {
    background: linear-gradient(90deg,#00d4aa,#0099ff) !important;
    color:#060910 !important; font-weight:800 !important;
    font-size:1.05rem !important; border:none !important;
    border-radius:12px !important; padding:0.7rem 2rem !important; width:100% !important;
}
h1,h2,h3,h4 { color:#c9d1d9; }
label { color:#6e7a8a !important; font-size:0.83rem !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────
FACTS = [
    " FICO score above 750 = excellent credit — lowest interest rates.",
    " Debt-to-Income (DTI) below 36% is considered safe by most lenders.",
    " Hard credit inquiries can lower FICO score by 5–10 points temporarily.",
    " Logistic Regression: σ(z) = 1/(1+e⁻ᶻ) converts score to probability.",
    " Class imbalance means defaulters are rare — models need special tuning.",
    " Global average loan default rate was ~3.5% in 2023.",
    " AUC-ROC above 0.70 = good credit risk model performance.",
    " Revolving utilization above 30% hurts credit score significantly.",
]

def detect_target(df):
    for c in ['not.fully.paid','default','loan_status','Default','charged_off','loan_default','target']:
        if c in df.columns: return c
    return df.columns[-1]

MAX_ROWS = 50000  # safe limit for most PCs

def prepare_df(df, tc):
    df = df.copy()
    # ── AUTO SAMPLE if too large ──
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        st.info(f" Large dataset detected! Auto-sampled to {MAX_ROWS:,} rows for performance.")
    # ── DROP high-cardinality columns (too many unique values = memory crash) ──
    for col in df.select_dtypes(include='object').columns:
        if col != tc and df[col].nunique() > 50:
            df = df.drop(columns=[col])
    for col in df.select_dtypes(include='object').columns:
        if col != tc:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    df = df.dropna()
    if df[tc].dtype == object:
        df[tc] = LabelEncoder().fit_transform(df[tc])
    return df

def train_model(df, tc):
    dfc = prepare_df(df, tc)
    X, y = dfc.drop(tc, axis=1), dfc[tc]
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    m  = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    m.fit(sc.fit_transform(Xtr), ytr)
    yp  = m.predict(sc.transform(Xte))
    ypr = m.predict_proba(sc.transform(Xte))[:,1]
    return m, sc, X.columns.tolist(), Xte, yte, yp, ypr

def loading_bar():
    bar = st.progress(0); tip = st.empty()
    for i,f in enumerate(FACTS):
        tip.markdown(f'<div class="tip"> Training model... {f}</div>', unsafe_allow_html=True)
        bar.progress((i+1)*12); time.sleep(0.22)
    bar.progress(100); tip.empty()

# ── HERO ────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title"> LoanGuard AI</div>
  <div class="hero-sub">Upload your loan dataset → Get instant ML predictions, graphs & risk analysis</div>
</div>
""", unsafe_allow_html=True)

# ── BIG UPLOAD BOX ──────────────────────────────
st.markdown("""
<div class="upload-box">
  <div class="upload-icon"></div>
  <div class="upload-title">Upload Your Loan Dataset</div>
  <div class="upload-sub">Supports: loan_data.csv · Home Credit · Lending Club · Credit Risk Dataset · Any loan CSV</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded:
    loading_bar()

    try:
        df = pd.read_csv(uploaded)
        tc = detect_target(df)
        m, sc, cols, Xte, yte, yp, ypr = train_model(df, tc)
        auc = roc_auc_score(yte, ypr)
        acc = (yp == yte).mean()

        # ── SUCCESS ──
        st.success(f" Model trained! &nbsp;|&nbsp; {df.shape[0]:,} rows × {df.shape[1]} cols &nbsp;|&nbsp; Target: `{tc}` &nbsp;|&nbsp; AUC: **{auc:.3f}** &nbsp;|&nbsp; Accuracy: **{acc:.1%}**")

        # ── DATASET OVERVIEW ──
        st.markdown('<div class="sec-head">// Dataset Overview</div>', unsafe_allow_html=True)
        default_rate = df[tc].value_counts(normalize=True).iloc[-1]
        o1,o2,o3,o4 = st.columns(4)
        o1.metric(" Total Rows",    f"{df.shape[0]:,}")
        o2.metric(" Features",      df.shape[1]-1)
        o3.metric(" Default Rate",  f"{default_rate:.1%}")
        o4.metric(" AUC Score",     f"{auc:.3f}")

        # ── OVERALL RISK VERDICT ──
        avg_prob = ypr.mean()
        st.markdown('<div class="sec-head">// Portfolio Risk Verdict</div>', unsafe_allow_html=True)

        if avg_prob < 0.30:
            rcss,rcolor,remoji,rlabel,radvice,rbadge = \
                "result-low","#00d4aa","","LOW RISK PORTFOLIO",\
                "This dataset shows strong overall creditworthiness. Low default probability across borrowers.",\
                "background:#00d4aa22;color:#00d4aa"
        elif avg_prob < 0.50:
            rcss,rcolor,remoji,rlabel,radvice,rbadge = \
                "result-mid","#ffaa00","","MODERATE RISK PORTFOLIO",\
                "Mixed risk profile. Some borrowers may need additional verification before approval.",\
                "background:#ffaa0022;color:#ffaa00"
        else:
            rcss,rcolor,remoji,rlabel,radvice,rbadge = \
                "result-high","#ff3333","","HIGH RISK PORTFOLIO",\
                "High default probability detected. This dataset contains many risky borrowers.",\
                "background:#ff333322;color:#ff3333"

        bar_w = int(avg_prob*100)
        st.markdown(f"""
        <div class="result-wrap {rcss}">
            <div class="r-emoji">{remoji}</div>
            <div class="r-label" style="color:{rcolor}">{rlabel}</div>
            <div class="r-pct"   style="color:{rcolor}">{avg_prob:.1%}</div>
            <div class="r-sub">Average Default Probability Across All Borrowers</div>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{bar_w}%;background:{rcolor}"></div>
            </div>
            <div class="r-advice" style="{rbadge}">{radvice}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── MODEL ACCURACY CARDS ──
        st.markdown('<div class="sec-head">// Prediction Accuracy Metrics</div>', unsafe_allow_html=True)
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(yte, yp, zero_division=0)
        rec  = recall_score(yte, yp, zero_division=0)
        f1   = f1_score(yte, yp, zero_division=0)

        a1,a2,a3,a4,a5 = st.columns(5)
        a1.metric(" Accuracy",  f"{acc:.1%}")
        a2.metric(" AUC-ROC",   f"{auc:.3f}")
        a3.metric(" Precision", f"{prec:.1%}")
        a4.metric(" Recall",    f"{rec:.1%}")
        a5.metric(" F1 Score",  f"{f1:.3f}")

        # ── GRAPHS ──
        st.markdown('<div class="sec-head">// Data & Model Visualizations</div>', unsafe_allow_html=True)

        bg="#060910"; ac="#00d4aa"
        plt.rcParams.update({
            'figure.facecolor':bg,'axes.facecolor':bg,
            'text.color':'white','axes.labelcolor':'#8892a4',
            'xtick.color':'#8892a4','ytick.color':'#8892a4',
            'axes.titlesize':13,'axes.titlepad':12,
        })

        g1,g2 = st.columns(2)

        # GRAPH 1 — Class Distribution
        with g1:
            st.markdown('<div class="g-title"> Default vs No Default Distribution</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            counts = df[tc].value_counts().sort_index()
            labels = ['No Default','Default'][:len(counts)]
            clrs   = [ac,'#ff3333'][:len(counts)]
            bars   = ax.bar(labels, counts.values, color=clrs, width=0.5, edgecolor='none')
            for bar,val in zip(bars,counts.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+counts.max()*0.02,
                        f'{val:,}',ha='center',color='white',fontsize=12,fontweight='bold')
            ax.set_title('Class Distribution'); ax.set_ylabel('Count')
            for sp in ax.spines.values(): sp.set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        # GRAPH 2 — ROC Curve
        with g2:
            st.markdown('<div class="g-title"> ROC Curve (AUC = Model Power)</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            fpr,tpr,_ = roc_curve(yte,ypr)
            ax.plot(fpr,tpr,color=ac,lw=2.5,label=f'AUC = {auc:.3f}')
            ax.plot([0,1],[0,1],'w--',lw=1,alpha=0.3,label='Random (0.5)')
            ax.fill_between(fpr,tpr,alpha=0.12,color=ac)
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(facecolor='#0d1117',labelcolor='white',fontsize=10)
            for sp in ax.spines.values(): sp.set_color('#21262d')
            st.pyplot(fig, use_container_width=True); plt.close()

        g3,g4 = st.columns(2)

        # GRAPH 3 — Confusion Matrix
        with g3:
            st.markdown('<div class="g-title"> Confusion Matrix</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            cm = confusion_matrix(yte,yp)
            sns.heatmap(cm,annot=True,fmt='d',cmap='YlOrRd',ax=ax,
                        linewidths=1,linecolor=bg,
                        annot_kws={"size":16,"weight":"bold","color":"white"},
                        xticklabels=['No Default','Default'],
                        yticklabels=['No Default','Default'])
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig, use_container_width=True); plt.close()

        # GRAPH 4 — Feature Importance
        with g4:
            st.markdown('<div class="g-title"> Top Features Driving Default</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,4))
            imp = pd.Series(np.abs(m.coef_[0]),index=cols).sort_values(ascending=True).tail(8)
            clrs_f = [ac if v>imp.median() else '#1a6b5a' for v in imp.values]
            ax.barh(imp.index,imp.values,color=clrs_f,edgecolor='none',height=0.6)
            ax.set_title('Feature Importance (|Coefficient|)')
            ax.set_xlabel('Importance Score')
            for sp in ax.spines.values(): sp.set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        # GRAPH 5 — Probability Distribution (full width)
        st.markdown('<div class="g-title"> Predicted Default Probability Distribution</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(12,4))
        ax.hist(ypr[yte==0],bins=50,alpha=0.75,color=ac,label='No Default',edgecolor='none')
        ax.hist(ypr[yte==1],bins=50,alpha=0.75,color='#ff3333',label='Default',edgecolor='none')
        ax.axvline(0.5,color='white',ls='--',lw=2,alpha=0.7,label='Decision Threshold (0.5)')
        ax.set_xlabel('Default Probability'); ax.set_ylabel('Number of Borrowers')
        ax.set_title('How well does the model separate defaulters from non-defaulters?')
        ax.legend(facecolor='#0d1117',labelcolor='white',fontsize=11)
        for sp in ax.spines.values(): sp.set_color('#21262d')
        st.pyplot(fig, use_container_width=True); plt.close()

        # GRAPH 6 — Target Feature Correlations
        st.markdown('<div class="g-title"> Feature Correlation with Default (Target Analysis)</div>',
                    unsafe_allow_html=True)
        dfc2  = prepare_df(df, tc)
        corrs = dfc2.corr()[tc].drop(tc).sort_values()
        top_corrs = pd.concat([corrs.head(6), corrs.tail(6)])
        fig,ax = plt.subplots(figsize=(12,4))
        colors_c = ['#ff3333' if v>0 else ac for v in top_corrs.values]
        ax.barh(top_corrs.index, top_corrs.values, color=colors_c, edgecolor='none', height=0.6)
        ax.axvline(0,color='white',lw=0.8,alpha=0.3)
        ax.set_title(f'Feature Correlation with Target ({tc})')
        ax.set_xlabel('Correlation Coefficient')
        for sp in ax.spines.values(): sp.set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── BATCH PREDICTIONS ──
        st.markdown('<div class="sec-head">// Individual Borrower Risk Predictions</div>',
                    unsafe_allow_html=True)
        Xa = dfc2.drop(tc,axis=1)
        for c in cols:
            if c not in Xa.columns: Xa[c] = 0
        Xa     = Xa[cols]
        all_p  = m.predict_proba(sc.transform(Xa))[:,1]

        result_df = df.copy()
        result_df[' Default Probability'] = [f"{p:.1%}" for p in all_p]
        result_df[' Risk Level'] = [
            ' HIGH'   if p>0.6  else
            ' MEDIUM' if p>0.35 else
            ' LOW'    for p in all_p]

        high = sum(1 for p in all_p if p>0.6)
        mid  = sum(1 for p in all_p if 0.35<=p<=0.6)
        low  = sum(1 for p in all_p if p<0.35)

        r1,r2,r3 = st.columns(3)
        r1.metric(" High Risk Borrowers",   f"{high:,}")
        r2.metric(" Medium Risk Borrowers", f"{mid:,}")
        r3.metric(" Low Risk Borrowers",    f"{low:,}")

        st.dataframe(result_df.head(100), use_container_width=True)

        st.download_button(" Download All Predictions CSV",
                            result_df.to_csv(index=False).encode('utf-8'),
                            "loan_predictions.csv","text/csv")

        # ── LEGAL FOOTER ──
        st.markdown('<div class="sec-head">// Legal & Regulatory Notice</div>', unsafe_allow_html=True)
        for n in [
            " EDUCATIONAL USE ONLY — Not financial or legal advice.",
            " No data stored or sent anywhere. All predictions run locally on your machine.",
            " Real lending in India must comply with RBI Fair Practices Code.",
            " ML predictions must NOT be the sole basis for any real lending decision.",
        ]:
            st.markdown(f'<div class="lcard">{n}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f" Error: {e}")
        st.info("Make sure your CSV has a target column named: default / loan_status / not.fully.paid")

else:
    # Show supported datasets while waiting
    st.markdown('<div class="sec-head">// What You\'ll Get After Upload</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        for f in [
            " Class distribution graph",
            " ROC Curve — model power",
            " Confusion Matrix — error types",
            " Feature importance chart",
            " Probability distribution graph",
            " Feature correlation with target",
        ]:
            st.markdown(f'<div class="kcard">{f}</div>', unsafe_allow_html=True)
    with c2:
        for f in [
            " Accuracy, AUC, Precision, Recall, F1",
            " Risk verdict — LOW / MEDIUM / HIGH",
            " Per-borrower risk predictions",
            " Download predictions as CSV",
            " ML knowledge during loading",
            " Legal & regulatory notes",
        ]:
            st.markdown(f'<div class="kcard">{f}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#2d3440;font-size:0.72rem;
margin-top:3rem;font-family:'IBM Plex Mono',monospace;">
LOANGUARD AI · ML PORTFOLIO PROJECT · EDUCATIONAL USE ONLY · NOT FINANCIAL ADVICE
</div>

""", unsafe_allow_html=True)

