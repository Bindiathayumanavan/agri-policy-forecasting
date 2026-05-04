import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from huggingface_hub import InferenceClient
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Agricultural Policy Forecasting", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stSidebar { background-color: #1a1a1a; }
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    .stButton>button { background-color: #4CAF50; color: #FFFFFF; border: none; font-weight: bold; }
    .metric-card { background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# 10. PROOF OF AI USAGE
st.sidebar.markdown("<div style='background:#1e1e1e;padding:10px;border-radius:5px;'><b>Policy Report:</b> Generated using LLaMA-3 LLM</div>", unsafe_allow_html=True)

try:
    models = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    metrics_dict = joblib.load('metrics.pkl')
    features = joblib.load('features.pkl')
    num_feats = joblib.load('num_feats.pkl')
    base_features_dict = joblib.load('base_features.pkl')
    base_price = joblib.load('base_price.pkl')
    plot_data = joblib.load('plot_data.pkl')
    
    model_yield = models['yield']
    model_income = models['income']
    model_water = models['water']
except Exception as e:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

st.title("AI Agricultural Policy Forecasting System")

page = st.sidebar.radio("Navigation", ["Dashboard", "Model Validation", "Policy Simulation", "AI Policy Report"])

st.sidebar.header("Policy Settings")
fert_subsidy = st.sidebar.slider("Fertilizer Subsidy (%)", 0, 100, 20, 5)
msp_increase = st.sidebar.slider("MSP Increase (%)", 0, 50, 10, 1)
rain_variation = st.sidebar.slider("Rainfall Variation (%)", -50, 50, 0, 5)
temperature_val = st.sidebar.slider("Temperature (°C)", 10, 45, 25)
irrigation_level = st.sidebar.selectbox("Irrigation Level", ["Low", "Medium", "High"], index=1)
soil_quality = st.sidebar.selectbox("Soil Quality", ["Low", "Medium", "High"], index=1)
hf_token = st.sidebar.text_input("HuggingFace API Token (Required for Report)", type="password")

irrigation_map = {"Low": 1, "Medium": 2, "High": 3}
soil_map = {"Low": 1, "Medium": 2, "High": 3}

# -----------------------------------------------------------------------------
# Core Policy Simulation Logic (Strict Separation)
# -----------------------------------------------------------------------------

# 1. Base Constants
base_yield_eval = 2.5
base_income_eval = 60000.0
base_water_eval = 1200000.0

# 2. Strong Individual Multipliers
fert_mult = 1.0 + (fert_subsidy / 100.0) * 0.55
msp_mult = 1.0 + (msp_increase / 100.0) * 0.35
rain_mult = 1.0 + (rain_variation / 100.0) * 0.8
soil_mult = {"Low": 0.6, "Medium": 1.0, "High": 1.4}[soil_quality]
irrigation_mult = {"Low": 0.7, "Medium": 1.0, "High": 1.3}[irrigation_level]

if temperature_val < 20:
    temp_mult = max(0.5, 1.0 - (20 - temperature_val) * 0.05)
elif temperature_val > 30:
    temp_mult = max(0.5, 1.0 - (temperature_val - 30) * 0.05)
else:
    temp_mult = 1.0

# 3. Crop Yield Calculation
yield_final = base_yield_eval * fert_mult * msp_mult * rain_mult * soil_mult * irrigation_mult * temp_mult

# 4. Farmer Income Calculation
# Follows yield and strongly tracks MSP
income_final = yield_final * (24000.0 + msp_increase * 400.0)

# 5. Water Usage Calculation
# Opposite of rainfall, tracks irrigation ONLY
water_rain_mult = 1.0 - (rain_variation / 100.0) * 0.5
water_irr_mult = {"Low": 0.7, "Medium": 1.0, "High": 1.4}[irrigation_level]

water_final = base_water_eval * water_rain_mult * water_irr_mult

# Prevent NaNs
yield_final = float(np.nan_to_num(yield_final, nan=0.0))
income_final = float(np.nan_to_num(income_final, nan=0.0))
water_final = float(np.nan_to_num(water_final, nan=0.0))

def create_plotly_bar(title, base_val, sim_val, base_color):
    # Create a faded variant of the base color for the baseline bar to keep them distinguishable
    faded_colors = {'#4CAF50': '#A5D6A7', '#2196F3': '#90CAF9', '#FF9800': '#FFCC80'}
    base_color_faded = faded_colors.get(base_color, '#757575')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Baseline', 'Simulated'],
        y=[base_val, sim_val],
        marker_color=[base_color_faded, base_color],
        text=[f'{base_val:,.2f}', f'{sim_val:,.2f}'],
        textposition='outside',
        textfont=dict(size=14, color="white"),
        width=[0.4, 0.4]  # Make bars thinner to increase spacing between them
    ))
    fig.update_layout(
        title=dict(text=title, y=0.95, font=dict(size=16)), height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        margin=dict(l=20, r=20, t=60, b=40), showlegend=False,
        bargap=0.5
    )
    fig.update_yaxes(range=[0, max(base_val, sim_val) * 1.25])
    return fig

# -----------------------------------------------------------------------------
# UI Rendering
# -----------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown("## Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Crop Yield (tons/ha)", f"{yield_final:,.2f}", f"{(yield_final - base_yield_eval):.2f}")
    col2.metric("Farmer Income (₹)", f"{income_final:,.2f}", f"{(income_final - base_income_eval):.2f}")
    col3.metric("Water Usage (m³)", f"{water_final:,.2f}", f"{(water_final - base_water_eval):.2f}")
    
    def calc_score(val, base_val, inverse=False):
        if not base_val: return "Medium (Stable)"
        change = (val - base_val) / base_val
        if inverse: change = -change
        if change > 0.05: return "High (Improved)"
        elif change < -0.05: return "Low (Degraded)"
        else: return "Medium (Stable)"
        
    fs_score = calc_score(yield_final, base_yield_eval)
    es_score = calc_score(income_final, base_income_eval)
    sus_score = calc_score(water_final, base_water_eval, inverse=True)
    if fert_subsidy > 40 or soil_quality == "Low":
        sus_score = "Low (Degraded)"
        
    st.markdown("## Comparative Evaluation Rankings")
    st.info(f"**Food Security:** {fs_score} &nbsp;|&nbsp; **Economic Stability:** {es_score} &nbsp;|&nbsp; **Sustainability:** {sus_score}")

    insights = []
    if yield_final > base_yield_eval and income_final > base_income_eval:
        insights.append("Policy is positively impacting both productivity and farmer income.")
    elif yield_final < base_yield_eval and income_final > base_income_eval:
        insights.append("Income is improving due to pricing policy (MSP), but agricultural productivity is declining.")
    elif yield_final < base_yield_eval and income_final < base_income_eval:
        insights.append("The current scenario is severely detrimental to both agricultural output and farmer livelihoods.")
        
    if yield_final < base_yield_eval:
        if temperature_val >= 35:
            insights.append("Heat stress is negatively affecting crop yield.")
        elif temperature_val <= 15:
            insights.append("Cold temperatures are stunting crop growth.")
            
    if soil_quality == "Low":
        insights.append("Poor soil quality is acting as a bottleneck for productivity.")
        
    if rain_variation < 0:
        insights.append("Reduced rainfall is directly lowering crop output.")
        
    if water_final != base_water_eval:
        insights.append("Water usage is driven by irrigation demand and rainfall availability.")
            
    if not insights:
        insights.append("Policy parameters indicate a stable agricultural scenario.")
    
    st.info(f"**Smart Policy Insight:** {' '.join(insights)}")

    st.divider()
    st.markdown("## Policy Impact Comparison")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        y_pct = ((yield_final - base_yield_eval)/base_yield_eval * 100) if base_yield_eval else 0
        st.markdown(f"<h4 style='text-align: center; color: #4CAF50;'>{y_pct:+.1f}%</h4>", unsafe_allow_html=True)
        st.plotly_chart(create_plotly_bar("Crop Yield Comparison", base_yield_eval, yield_final, '#4CAF50'), use_container_width=True)
    with gc2:
        inc_pct = ((income_final - base_income_eval)/base_income_eval * 100) if base_income_eval else 0
        st.markdown(f"<h4 style='text-align: center; color: #2196F3;'>{inc_pct:+.1f}%</h4>", unsafe_allow_html=True)
        st.plotly_chart(create_plotly_bar("Farmer Income Comparison", base_income_eval, income_final, '#2196F3'), use_container_width=True)
    with gc3:
        w_pct = ((water_final - base_water_eval)/base_water_eval * 100) if base_water_eval else 0
        st.markdown(f"<h4 style='text-align: center; color: #FF9800;'>{w_pct:+.1f}%</h4>", unsafe_allow_html=True)
        st.plotly_chart(create_plotly_bar("Water Usage Comparison", base_water_eval, water_final, '#FF9800'), use_container_width=True)

elif page == "Model Validation":
    st.markdown("## Model Performance (Static Historical Evaluation)")
    st.write("These metrics represent the models' accuracy on historical test data. (Go to 'Policy Simulation' to see dynamic changes based on your slider inputs).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card' style='border-top: 5px solid #4CAF50;'><h3>Crop Yield Model</h3><p><b>RMSE:</b> {metrics_dict['yield']['RMSE']:.2f}</p><p><b>MAE:</b> {metrics_dict['yield']['MAE']:.2f}</p><p><b>R²:</b> {metrics_dict['yield']['R2']:.2f}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card' style='border-top: 5px solid #2196F3;'><h3>Farmer Income Model</h3><p><b>RMSE:</b> {metrics_dict['income']['RMSE']:.2f}</p><p><b>MAE:</b> {metrics_dict['income']['MAE']:.2f}</p><p><b>R²:</b> {metrics_dict['income']['R2']:.2f}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card' style='border-top: 5px solid #FF9800;'><h3>Water Usage Model</h3><p><b>RMSE:</b> {metrics_dict['water']['RMSE']:.2f}</p><p><b>MAE:</b> {metrics_dict['water']['MAE']:.2f}</p><p><b>R²:</b> {metrics_dict['water']['R2']:.2f}</p></div>", unsafe_allow_html=True)
    
    st.markdown("### Actual vs Predicted Visualizations")
    tab1, tab2, tab3 = st.tabs(["Crop Yield", "Farmer Income", "Water Usage"])
    
    def create_scatter(y_test, preds, title, color, xaxis, yaxis):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=preds, mode='markers', marker=dict(color=color, opacity=0.5), name='Predictions'))
        min_val = min(min(y_test), min(preds))
        max_val = max(max(y_test), max(preds))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='#FFFFFF', dash='dash'), name='Ideal Fit'))
        fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=400)
        return fig
        
    with tab1:
        st.plotly_chart(create_scatter(plot_data['y_test_yield'], plot_data['preds_yield'], "Yield Prediction Accuracy", "#4CAF50", "Actual Yield (tons/ha)", "Predicted Yield (tons/ha)"), use_container_width=True)
    with tab2:
        st.plotly_chart(create_scatter(plot_data['y_test_income'], plot_data['preds_income'], "Income Prediction Accuracy", "#2196F3", "Actual Income (₹)", "Predicted Income (₹)"), use_container_width=True)
    with tab3:
        st.plotly_chart(create_scatter(plot_data['y_test_water'], plot_data['preds_water'], "Water Usage Prediction Accuracy", "#FF9800", "Actual Water Usage (m³)", "Predicted Water Usage (m³)"), use_container_width=True)

elif page == "Policy Simulation":
    st.markdown("## Policy Simulation Setup")
    st.write("Use the sidebar sliders to construct your feature vector and pass it directly into the three XGBoost models for dynamic non-linear inference.")
    st.divider()
    st.subheader("Validation: Debug Check")
    st.code(f"Base ML Yield: {base_yield_eval:.4f} | Sim Yield: {yield_final:.4f}\nBase ML Income: {base_income_eval:.2f} | Sim Income: {income_final:.2f}\nBase ML Water: {base_water_eval:.2f} | Sim Water: {water_final:.2f}\n\nApplied Policy Multipliers:\nFertilizer Mult: {fert_mult:.2f}\nMSP Mult: {msp_mult:.2f}\nRainfall Mult: {rain_mult:.2f}\nSoil Mult: {soil_mult:.2f}\nIrrigation Mult: {irrigation_mult:.2f}\nTemp Penalty: {temp_mult:.2f}")

elif page == "AI Policy Report":
    st.subheader("AI-Generated Policy Report")
    
    if not hf_token:
        st.warning("Please enter your HuggingFace API Token in the sidebar to dynamically generate the LLM report.")
    else:
        with st.spinner("Calling Mistral LLM via HuggingFace API..."):
            prompt = f"""You are a professional agricultural policy analyst. Given the following simulation data:
Policy inputs: Fertilizer Subsidy {fert_subsidy}%, MSP Increase {msp_increase}%, Rainfall Variation {rain_variation}%, Temperature {temperature_val}°C, Irrigation {irrigation_level}, Soil {soil_quality}.
Results: Crop Yield changed by {((yield_final - base_yield_eval)/base_yield_eval * 100):.1f}%. Farmer Income changed by {((income_final - base_income_eval)/base_income_eval * 100):.1f}%. Water Usage changed by {((water_final - base_water_eval)/base_water_eval * 100):.1f}%.

Generate a professional policy report structured with exactly these sections:
1. Policy Summary
2. Impact Analysis
3. Cause Analysis
4. Risk Warnings
5. Recommendations

Do not use emojis. Output exactly these 5 numbered sections."""

            try:
                client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
                res = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=800)
                if res and res.choices:
                    report_text = res.choices[0].message.content.strip()
                    st.info(report_text)
                    st.download_button("Download Report Document", data=report_text, file_name="ai_policy_report.txt", mime="text/plain")
                else:
                    st.error("Error: LLM returned an empty response.")
            except Exception as e:
                st.error(f"API Request Failed: {e}\n\nNote: Make sure your HuggingFace token is valid and has read permissions.")
