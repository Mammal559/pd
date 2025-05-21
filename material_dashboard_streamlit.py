import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import altair as alt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
st.set_page_config(page_title="Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

# Inject Material Dashboard CSS, JS, and responsive styling
st.markdown("""
<!-- Fonts and Icons -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free/css/all.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/nucleo-icons/css/nucleo-icons.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/nucleo-svg/css/nucleo-svg.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/gh/creativetimofficial/material-dashboard@main/assets/css/material-dashboard.css?v=3.2.0" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/gh/creativetimofficial/material-dashboard@main/assets/js/material-dashboard.min.js?v=3.2.0"></script>

<style>
.block-container {
        padding-top: 1.3rem;
    }           
body {
    background-color:#f8f9fa;
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    
}
.stApp {
        background-color:#f8f9fa;            
 .kpi-card {
            background-color: #ffffff;
            border: 2px solid #adb5bd;  /* border with your hex */
            border-radius: 12px;
            padding: 6px;
            min-height: 70px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .kpi-card h6 {
            margin-bottom: 5px;
            font-size: 16px;
            color: #495057;
        }
        .kpi-card h3 {
            margin: 0;
            font-size: 20px;
            font-weight: 600;
            color: #212529;
        }
h1, h2, h3 {
    color: #344767;
    font-weight: 600;
}
.sidebar .sidebar-content {
    background: #495057;
    color: #495057;
    font-weight: 600;
    padding: 1rem;
    border-radius: 10px;
}
button.stButton > button {
    background-color: #4caf50;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease-in-out;
    margin-top: 1rem;
}
button.stButton > button:hover {
    background-color: #388e3c;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
/* Ensure full responsiveness on all screens */
.reportview-container .main .block-container {
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}
.plot-container.plotly {
    margin-bottom: 10px !important;
    border:2px solid
    solid #ccc;
            
}

            
</style>
""", unsafe_allow_html=True)



# Load data
df = pd.read_csv("enhanced_sales_marketing_dataset.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Rename and remove minus sign from performance_vs_target
df['achievd_target'] = df['performance_vs_target']
df.drop(columns=['performance_vs_target'], inplace=True)




# Sidebar Navigation
# Sidebar Navigation
with st.sidebar:
    st.image("https://demos.creative-tim.com/material-dashboard/assets/img/logo-ct-dark.png", width=120)
    st.markdown("Sales Dashboard")

    # Dropdown for view selection
    view = st.selectbox("Select View", ["Main View", "Sales View", "Marketing View", "Manager View", "Model View"])

    st.markdown("---")

    # Filter options
    device_filter = st.selectbox("Device Type", options=["All"] + sorted(df['device_type'].unique()))
    country_filter = st.selectbox("Country", options=["All"] + sorted(df['country'].unique()))
    sales_filter = st.selectbox("Sales Officer", options=["All"] + sorted(df['sales_person'].unique()))

    st.markdown("### ⏱ Hour Range Filter")
    start_hour = st.slider("Start Hour", 0, 23, 8)
    end_hour = st.slider("End Hour", 0, 23, 18)
    if start_hour > end_hour:
        st.warning("Start hour must be before end hour.")

    st.button("Apply Filters")

# Apply filters
filtered_df = df.copy()

if device_filter != "All":
    filtered_df = filtered_df[filtered_df['device_type'] == device_filter]
if country_filter != "All":
    filtered_df = filtered_df[filtered_df['country'] == country_filter]
if sales_filter != "All":
    filtered_df = filtered_df[filtered_df['sales_person'] == sales_filter]

# Filter by hour
filtered_df['hour'] = filtered_df['timestamp'].dt.hour
filtered_df = filtered_df[(filtered_df['hour'] >= start_hour) & (filtered_df['hour'] <= end_hour)]
filtered_df.drop(columns='hour', inplace=True)  # Optional cleanup

# MAIN VIEW
if view == "Main View":
    st.title("📊 Main Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div class='kpi-card'><h6>💰 Total Sales</h6><h3>${filtered_df['purchase_amount_usd'].sum():,.2f}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card'><h6>📦 Top Product</h6><h3>{filtered_df['product'].value_counts().idxmax()}</h3></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card'><h6>👥 Total Visitors</h6><h3>{len(filtered_df)}</h3></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card'><h6>🔁 Returning Users</h6><h3>{filtered_df['returning_customer'].sum()}</h3></div>", unsafe_allow_html=True)
    
    # Add vertical space
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)



    product_counts = filtered_df['product'].value_counts().reset_index()
    product_counts.columns = ['product', 'count']
    fig_product = px.bar(product_counts, x='product', y='count', color='product', title='Top Performing Products')
    fig_product.update_layout(height=300)

    
    fig_channel = px.pie(filtered_df, names='marketing_channel', title='Marketing Channel Distribution')
    fig_channel.update_layout(height=300)

    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    hourly_sales = filtered_df.groupby('hour')['purchase_amount_usd'].sum().reset_index()
    fig_hour = px.line(hourly_sales, x='hour', y='purchase_amount_usd', title='Sales by Hour', markers=True)
    fig_hour.update_layout(height=300)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_product, use_container_width=True)
    c2.plotly_chart(fig_channel, use_container_width=True)
    st.plotly_chart(fig_hour, use_container_width=True)

# Model View


elif view == "Model View":
    st.title("🤖 Enhanced Machine Learning Model for Salesperson Performance")

    # Select relevant features (customize as needed)
    model_df = df[['previous_visits', 'session_duration_minutes', 'conversion_history',
                   'device_type', 'marketing_channel', 'country', 'sales_person',
                   'purchase_amount_usd', 'returning_customer', 'achievd_target']].dropna()
    

    # Features & target
    categorical_features = ['device_type', 'marketing_channel', 'country', 'sales_person']
    numerical_features = ['previous_visits', 'session_duration_minutes', 'conversion_history',
                          'purchase_amount_usd', 'returning_customer']

    X = model_df[categorical_features + numerical_features]
    y = model_df['achievd_target']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # numerical features stay as-is
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline with preprocessing + XGBoost regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_estimators=100, max_depth=5))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring='r2').mean()

    # Display metrics
    st.markdown("### 📊 Model Evaluation Metrics")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Cross-validated R² (5-Fold):** {cv_score:.4f}")

    # Feature importance
    st.markdown("### 🔍 Feature Importances")
    model = pipeline.named_steps['regressor']
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model.feature_importances_

    #fig, ax = plt.subplots(figsize=(8, 6))
    #ax.barh(feature_names, importances)
    #ax.set_title("Feature Importance")
    #plt.tight_layout()
    #st.pyplot(fig)

    # --- Prediction UI ---
    st.markdown("---")
    st.markdown("### 🎯 Predict Salesperson Performance")

    input_col, gauge_col = st.columns([2, 1])

    with input_col:
        col1, col2 = st.columns(2)
        visits = col1.slider("Average Previous Visits", 0, 20, 5)
        duration = col2.slider("Average Session Duration (minutes)", 0, 60, 15)

        col3, col4 = st.columns(2)
        conversions = col3.slider("Average Past Conversions", 0, 10, 2)
        purchase_amount = col4.number_input("Average Purchase Amount (USD)", min_value=0.0, value=100.0)

        col5, col6 = st.columns(2)
        device_input = col5.selectbox("Device Type", df['device_type'].dropna().unique())
        marketing_input = col6.selectbox("Marketing Channel", df['marketing_channel'].dropna().unique())

        country_input = st.selectbox("Country", df['country'].dropna().unique())
        sales_person_input = st.selectbox("Sales Person", df['sales_person'].dropna().unique())

        returning_customer_input = st.selectbox("Returning Customer Rate", [0, 1], index=1)

        # Prepare input DataFrame
        user_input = pd.DataFrame([{
            'previous_visits': visits,
            'session_duration_minutes': duration,
            'conversion_history': conversions,
            'purchase_amount_usd': purchase_amount,
            'device_type': device_input,
            'marketing_channel': marketing_input,
            'country': country_input,
            'sales_person': sales_person_input,
            'returning_customer': returning_customer_input
        }])

        # Predict
        performance_pred = pipeline.predict(user_input)[0]
        st.markdown(f"#### ✅ Achieved Target: **{performance_pred:.2f}**")

    # --- Gauge chart ---
    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=performance_pred,
            title={'text': "Performance vs Target"},
            gauge={
                'axis': {'range': [0, max(150, performance_pred*1.5)]},
                'bar': {'color': "#636EFA"},
                'steps': [
                    {'range': [0, 80], 'color': "#FFDDDD"},
                    {'range': [80, 100], 'color': "#FFF7CC"},
                    {'range': [100, 150], 'color': "#D4EDDA"},
                ]
            }
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)


elif view == "Sales View":
    st.title("📊 Sales Performance")

    # --- DATA PROCESSING ---
    filtered_df['targets'] = (filtered_df['sales_target'] - filtered_df['purchase_amount_usd']).round(2)

    def status(row):
        if row['targets'] <= 0:
            return "✅ Achieved"
        elif row['targets'] <= 0.25 * row['sales_target']:
            return "🟡 On Track"
        elif row['targets'] <= 0.6 * row['sales_target']:
            return "🟠 At Risk"
        else:
            return "🔴 Critical"

    filtered_df['sales_status'] = filtered_df.apply(status, axis=1)

    sales_summary = filtered_df.groupby('sales_person').agg({
        'purchase_amount_usd': lambda x: round(x.sum(), 2),
        'sales_target': lambda x: round(x.sum(), 2),
        'targets': lambda x: round(x.sum(), 2),
        'country': lambda x: x.value_counts().idxmax(),
        'sales_status': lambda x: x.value_counts().idxmax()
    }).reset_index()

    total_sales = round(filtered_df['purchase_amount_usd'].sum(), 2)
    avg_target_gap = round(filtered_df['targets'].mean(), 2)
    target_hit_rate = round((filtered_df['sales_status'] == '✅ Achieved').mean() * 100, 1)

    # --- TOP BLOCK: KPIs ---
    with st.container():
         st.subheader("🧾 Individual Salesperson Performance")

    for i in range(0, len(sales_summary), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(sales_summary):
                row = sales_summary.iloc[i + j]
                person = row['sales_person']
                actual = row['purchase_amount_usd']
                target = row['sales_target']
                gap = target - actual
                status = row['sales_status']
                country = row['country']

                # Arrow + note
                if gap <= 0:
                    arrow = "🟢 ▲"
                    note = f"Exceeded by ${abs(gap):,.2f}"
                    gap_color = "#4CAF50"
                    progress_color = "#4CAF50"  # Green for achieved
                else:
                    arrow = "🔴 ▼"
                    note = f"Below target by ${gap:,.2f}"
                    gap_color = "#f44336"
                    progress_color = "#2196F3"  # Blue for in progress

                progress_pct = min(100, round((actual / target) * 100)) if target != 0 else 0

                with cols[j]:
                    st.markdown(f"""
                        <div style="background: #ffffff; padding: 20px; border-radius: 12px; 
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;
                                    border-left: 6px solid {gap_color};">
                            <h4 style="color: #333;">{person}</h4>
                            <p style="margin: 0;"><strong>🌍 Country:</strong> {country}</p>
                            <p style="margin: 8px 0 4px;"><strong>🎯 Target:</strong> ${target:,.2f}</p>
                            <p style="margin: 0;"><strong>💰 Actual:</strong> ${actual:,.2f}</p>
                            <div style="margin: 10px 0;">
                                <div style="height: 12px; background: #eee; border-radius: 6px; overflow: hidden;">
                                    <div style="width: {progress_pct}%; background: {progress_color}; height: 100%;"></div>
                                </div>
                                <small style="color: #666;">Progress: {progress_pct}%</small>
                            </div>
                            <p style="margin: 8px 0;">{arrow} <strong>{note}</strong></p>
                            <p style="margin-top: 10px;">📊 <strong>Status:</strong> {status}</p>
                        </div>
                    """, unsafe_allow_html=True)

    # --- MIDDLE BLOCK: MANAGER & SALES SIDE-BY-SIDE ---
    with st.container():
     left, right = st.columns(2)

    # 🌟 ALT-AIR CHART BLOCK (Left Side)
    with left:
        st.markdown("### 📈 Sales Target vs Actual (Altair)")
        line_data = sales_summary[['sales_person', 'sales_target', 'purchase_amount_usd']].melt(
            id_vars='sales_person',
            value_vars=['sales_target', 'purchase_amount_usd'],
            var_name='Metric',
            value_name='Amount'
        )

        line_chart = alt.Chart(line_data).mark_line(point=alt.OverlayMarkDef(color='black')).encode(
            x=alt.X('sales_person:N', title='Sales Person', sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Amount:Q', title='Amount (USD)'),
            color=alt.Color('Metric:N', title='', scale=alt.Scale(scheme='category10')),
            tooltip=['sales_person:N', 'Metric:N', alt.Tooltip('Amount:Q', format='$,.2f')]
        ).properties(
            title=alt.TitleParams(
                text='📉 Sales Target vs Actual Sales (Altair)',
                fontSize=16,
                font='Arial',
                anchor='start',
                color='#333'
            ),
            height=400
        ).configure_axis(
            grid=False
        ).configure_view(
            stroke=None
        )

        st.altair_chart(line_chart, use_container_width=True)

    # 🌟 PLOTLY VISUALIZATION BLOCK (Right Side)
    with right:
        st.markdown("### 📊 Actual vs Target Performance (Plotly)")

        # Prepare data
        plotly_summary = filtered_df.groupby('sales_person').agg({
            'purchase_amount_usd': 'sum',
            'sales_target': 'mean'
        }).reset_index()
        plotly_summary['Gap'] = plotly_summary['purchase_amount_usd'] - plotly_summary['sales_target']

        fig = go.Figure()

        # Actual Sales Bar
        fig.add_trace(go.Bar(
            x=plotly_summary['sales_person'],
            y=plotly_summary['purchase_amount_usd'],
            name='💰 Actual Sales',
            marker_color='rgba(38, 166, 154, 0.9)',
            text=[f"${x:,.0f}" for x in plotly_summary['purchase_amount_usd']],
            textposition='outside'
        ))

        # Sales Target Line
        fig.add_trace(go.Scatter(
            x=plotly_summary['sales_person'],
            y=plotly_summary['sales_target'],
            name='🎯 Target',
            mode='lines+markers',
            line=dict(color='rgba(244, 67, 54, 0.9)', width=3),
            marker=dict(symbol='circle', size=10)
        ))

        fig.update_layout(
            barmode='group',
            title="✨ Actual vs Target per Salesperson",
            xaxis=dict(title="Salesperson", tickangle=-45),
            yaxis=dict(title="Amount (USD)", tickformat="$.0f"),
            legend=dict(orientation='h', y=1.1, x=0),
            height=400,
            margin=dict(t=50, b=50),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)


    








# MANAGER VIEW
elif view == "Manager View":
    st.title("👔 Manager View")

    avg_sales = filtered_df['sales_target'].mean()
    avg_purchase = filtered_df['purchase_amount_usd'].mean()
    avg_conversion = filtered_df['conversion_rate'].mean()
    avg_engagement = filtered_df['engagement_score'].mean()

    metrics_df = pd.DataFrame({
        'Metric': ['Avg Sales Target', 'Avg Purchase', 'Avg Conversion Rate', 'Avg Engagement Score'],
        'Value': [avg_sales, avg_purchase, avg_conversion * 100, avg_engagement]
    })

    fig1 = px.bar(metrics_df, x='Metric', y='Value', color='Metric', title='Overall Averages',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig1.update_layout(height=350)

    top_sales = filtered_df.groupby('sales_person')['purchase_amount_usd'].sum().reset_index().nlargest(5, 'purchase_amount_usd')
    fig2 = px.bar(top_sales, x='sales_person', y='purchase_amount_usd', title='Top 5 Sales Performers',
                  color='purchase_amount_usd', color_continuous_scale='greens')
    fig2.update_layout(height=350)

    top_channels = filtered_df.groupby('marketing_channel')['conversion_rate'].mean().reset_index().nlargest(5, 'conversion_rate')
    fig3 = px.bar(top_channels, x='marketing_channel', y='conversion_rate', title='Top 5 Marketing Channels',
                  color='conversion_rate', color_continuous_scale='greens')
    fig3.update_layout(height=350)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
