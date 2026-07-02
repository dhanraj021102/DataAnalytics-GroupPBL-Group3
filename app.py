import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder  # FIX: moved to top-level imports

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Advanced Marketing Data Science", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #00D2D3; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1A1C24; padding: 15px; border-radius: 10px; border-left: 4px solid #00D2D3; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Advanced Marketing Data Science Framework")
st.markdown("Evaluating Data Quality, Segmentation, Predictive Models, and Actionable Prescriptive Analytics.")

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Disposable_Socks_Consumer_Data.xlsx", type=["csv", "xlsx", "xls"])

df_raw = load_data(uploaded_file)

if df_raw is None:
    st.warning("⚠️ Please upload the dataset to begin the analysis.")
    st.stop()

# Auto-detect targets for classification and regression
classification_target = 'WTP_Disposable_Socks' if 'WTP_Disposable_Socks' in df_raw.columns else df_raw.columns[-1]
regression_target = 'Monthly_Activewear_Spend' if 'Monthly_Activewear_Spend' in df_raw.columns else df_raw.select_dtypes(include=np.number).columns[-1]

# FIX: guard against both targets resolving to the same column when the
# named columns aren't present and the fallbacks collide.
if classification_target == regression_target:
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    alt_candidates = [c for c in numeric_cols if c != regression_target]
    if alt_candidates:
        classification_target = alt_candidates[-1]
    else:
        st.error("Could not find two distinct columns to use as classification and regression targets. "
                  "Please provide a dataset with at least two suitable target columns.")
        st.stop()

df_clean = df_raw.drop(columns=['Response_ID', 'Open_Ended_Reason'], errors='ignore').dropna()

# --- TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 1. Data Assessment",
    "🧩 2. Segmentation",
    "🔮 3. Classification",
    "📈 4. Spend Regression",
    "🛒 5. Association Rules",
    "🚀 6. Placement & Strategy"
])

# -----------------------------------------
# TAB 1: DATA ASSESSMENT & QUALITY
# -----------------------------------------
with t1:
    st.header("Data Quality & Sufficiency Assessment")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_raw.shape[0])
    col2.metric("Features", df_raw.shape[1])
    col3.metric("Missing Values", df_raw.isnull().sum().sum())
    col4.metric("Sufficiency", "High" if df_raw.shape[0] > 100 else "Low")

    st.subheader("Feature Types & Missingness")
    info_df = pd.DataFrame({
        'Data Type': df_raw.dtypes.astype(str),
        'Missing Values': df_raw.isnull().sum(),
        'Missing %': (df_raw.isnull().sum() / len(df_raw)) * 100
    })
    st.dataframe(info_df.T)

    st.subheader("Continuous vs Categorical & Outliers")
    # FIX: this selectbox was built from df_raw's numeric columns, but the
    # chart below plots df_clean, which has Response_ID/Open_Ended_Reason
    # dropped. Picking Response_ID (numeric) crashed px.box with a
    # "column not found in dataframe" ValueError. Build options from
    # df_clean so the dropdown and the chart are always in sync.
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        selected_num = st.selectbox("Select variable to check for outliers", num_cols)
        fig_box = px.box(df_clean, y=selected_num, title=f"Outlier Detection: {selected_num}", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns available in the cleaned dataset for outlier detection.")

    st.subheader("Class Balance (Target Variable)")
    if classification_target in df_clean.columns:
        fig_bal = px.histogram(df_clean, x=classification_target, color=classification_target, template="plotly_dark")
        st.plotly_chart(fig_bal, use_container_width=True)

# -----------------------------------------
# TAB 2: CUSTOMER SEGMENTATION (K-MEANS vs LCA)
# -----------------------------------------
with t2:
    st.header("Customer Segmentation (K-Means vs. GMM/LCA Proxy)")

    st.write("Comparing K-Means (distance-based) with Gaussian Mixture Models (probabilistic, proxy for Latent Class Analysis).")

    # FIX: exclude the prediction targets from the clustering feature set so
    # personas aren't partly defined by the outcomes we later try to predict.
    X_seg = df_clean.select_dtypes(include=np.number).drop(
        columns=[classification_target, regression_target], errors='ignore'
    ).copy()

    if X_seg.empty:
        st.error("No numeric columns available for clustering (after excluding target variables).")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_seg)

        n_clusters = st.slider("Select Number of Personas (Clusters)", 2, 6, 3)

        # K-Means
        # FIX: n_init='auto' requires scikit-learn >= 1.4; use an explicit
        # integer so this works across the whole >=1.2.0 range in requirements.txt.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        # GMM (LCA Proxy)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        df_seg = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'KMeans': kmeans_labels.astype(str), 'GMM': gmm_labels.astype(str)})

        c1, c2 = st.columns(2)
        with c1:
            fig_km = px.scatter(df_seg, x='PCA1', y='PCA2', color='KMeans', title="K-Means Segmentation", template="plotly_dark")
            st.plotly_chart(fig_km, use_container_width=True)
        with c2:
            fig_gmm = px.scatter(df_seg, x='PCA1', y='PCA2', color='GMM', title="Gaussian Mixture (LCA Proxy)", template="plotly_dark")
            st.plotly_chart(fig_gmm, use_container_width=True)

        st.info("💡 **Insight:** K-Means forces spherical clusters, making it highly interpretable for distinct spending tiers. GMM (LCA) allows for elliptical clusters, better capturing probabilistic overlaps in behavioral personas.")

# -----------------------------------------
# TAB 3: CLASSIFICATION (SPENDING READINESS)
# -----------------------------------------
with t3:
    st.header("Predictive Classification: Spending Readiness")
    st.write(f"Targeting: `{classification_target}` using RF and GBM.")

    X_clf = df_clean.drop(columns=[classification_target, regression_target], errors='ignore')
    y_clf = df_clean[classification_target]

    for col in X_clf.select_dtypes(include=['object', 'category']).columns:
        X_clf[col] = LabelEncoder().fit_transform(X_clf[col].astype(str))

    # FIX: a numeric target with many unique values (e.g. a 1-100 rating
    # scale) was previously left untouched and treated as a multi-class
    # classification problem with one class per value, producing a
    # near-meaningless model. Bin high-cardinality numeric targets instead.
    MAX_CLASSES = 10
    if y_clf.dtype == 'object':
        y_clf = LabelEncoder().fit_transform(y_clf)
    elif np.issubdtype(y_clf.dtype, np.number) and y_clf.nunique() > MAX_CLASSES:
        st.info(f"ℹ️ `{classification_target}` has {y_clf.nunique()} unique numeric values — "
                f"binning into {MAX_CLASSES} quantile-based classes for classification.")
        y_clf = pd.qcut(y_clf, q=MAX_CLASSES, duplicates='drop')
        y_clf = LabelEncoder().fit_transform(y_clf.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    scaler_clf = StandardScaler()
    X_train_s = scaler_clf.fit_transform(X_train)
    X_test_s = scaler_clf.transform(X_test)

    models_clf = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    res_clf, roc_data = [], []
    for name, model in models_clf.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1] if len(np.unique(y_clf)) == 2 else None

        res_clf.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data.append(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc(fpr, tpr):.2f})'))

    st.dataframe(pd.DataFrame(res_clf).style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score']))

    if roc_data:
        fig_roc = go.Figure(data=roc_data)
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(title="ROC-AUC Comparison", template="plotly_dark", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

# -----------------------------------------
# TAB 4: SPEND REGRESSION
# -----------------------------------------
with t4:
    st.header("Predictive Regression: Expected Monthly Spend")
    st.write(f"Targeting: `{regression_target}` to predict customer lifetime value/monthly revenue.")

    if regression_target not in df_clean.columns:
        st.warning("Regression target missing.")
    else:
        X_reg = df_clean.drop(columns=[regression_target, classification_target], errors='ignore')
        y_reg = df_clean[regression_target]

        for col in X_reg.select_dtypes(include=['object', 'category']).columns:
            X_reg[col] = LabelEncoder().fit_transform(X_reg[col].astype(str))

        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        # FIX: use a dedicated scaler for this tab instead of silently
        # reusing/refitting the classification tab's scaler_clf. The two
        # feature sets happen to line up today, but that's an implicit,
        # fragile coupling between tabs.
        scaler_reg = StandardScaler()
        X_tr_s = scaler_reg.fit_transform(X_tr)
        X_te_s = scaler_reg.transform(X_te)

        reg_models = {
            "Linear Regression": LinearRegression(),
            "Ridge (L2 Penalty)": Ridge(alpha=1.0),
            "Lasso (L1 Penalty)": Lasso(alpha=0.1)
        }

        res_reg = []
        for name, model in reg_models.items():
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            res_reg.append({
                "Model": name,
                "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
                "MAE": mean_absolute_error(y_te, preds),
                "R2 Score": r2_score(y_te, preds)
            })

        st.dataframe(pd.DataFrame(res_reg).style.highlight_max(axis=0, subset=['R2 Score']))
        st.info("💡 **Model Recommendation:** Lasso regression often performs best in marketing by zeroing out irrelevant features, making the final model highly interpretable for business stakeholders.")

# -----------------------------------------
# TAB 5: ASSOCIATION RULES (BASKET ANALYSIS)
# -----------------------------------------
with t5:
    st.header("Market Basket & Association Rules")
    st.write("Identifying product bundles and Next-Best-Offer (NBO).")

    # Generate Synthetic Basket Data for Demonstration (since standard df lacks basket lists)
    np.random.seed(42)
    transactions = []
    items = ['Disposable Socks', 'Gym Bag', 'Pre-Workout', 'Water Bottle', 'Towel']
    for _ in range(len(df_clean)):
        basket = np.random.choice(items, size=np.random.randint(1, 4), replace=False)
        # Induce a rule: if Gym Bag, likely Disposable Socks
        if 'Gym Bag' in basket and np.random.rand() > 0.3:
            if 'Disposable Socks' not in basket:
                basket = np.append(basket, 'Disposable Socks')
        transactions.append(basket)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_basket = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(df_basket, min_support=0.1, use_colnames=True)

    # FIX: `num_itemsets` was only added to association_rules() in
    # mlxtend 0.23.1; requirements.txt allowed mlxtend>=0.22.0, which would
    # raise TypeError: unexpected keyword argument 'num_itemsets'.
    # Call it in a version-tolerant way and pin the minimum version below.
    try:
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0, num_itemsets=len(df_basket))
    except TypeError:
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)

    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10))
    else:
        st.write("No strong rules found with current thresholds.")

# -----------------------------------------
# TAB 6: PLACEMENT & STRATEGY
# -----------------------------------------
with t6:
    st.header("Customer Placement Framework & Business Insights")

    st.markdown("""
    ### 🎯 Customer Placement Matrix
    | Segment | Persona | Expected Spend | Readiness | Recommended NBO (Next Best Offer) |
    |---|---|---|---|---|
    | **Tier 1** | The Prepared Athlete | High ($80+) | Low (Has own gear) | Premium supplements, Bulk discounts |
    | **Tier 2** | The Rushed Professional | Medium ($40-$80) | **High (Forgets gear)** | **Disposable Socks**, Hygiene Wipes |
    | **Tier 3** | The Casual Gym-Goer | Low (<$40) | Medium | Entry-level gym bags, Single-use items |

    ### 🧠 Data Science Insights & Deployments
    1. **Data Sufficiency:** The dataset contains excellent behavioral flags. Continuous variables (like Age or Spend) are best kept continuous for Regression but should be binned (categorized) for Association Rule Mining.
    2. **Model Choice:**
       - *Classification:* **Gradient Boosting** generally outperforms RF in detecting subtle behavioral cues for "Spending Readiness".
       - *Regression:* **Ridge Regression** handles multicollinearity well, but **Lasso** provides a cleaner, sparse model that highlights the true drivers of Monthly Spend.
    3. **Actionable Strategy:** Deploy the Gradient Boosting model in the backend of the gym app. When a "Rushed Professional" persona checks into the gym via Bluetooth beacon, trigger a push notification for Disposable Socks (Confidence > 65% based on Apriori rules).
    """)
