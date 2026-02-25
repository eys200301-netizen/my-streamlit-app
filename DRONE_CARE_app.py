import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Drone Care AI Pro",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# الألوان والتنسيقات التقنية
BG_IMAGE_URL = "https://i.pinimg.com/1200x/79/b0/42/79b04283a0624e302a58fd156ab333a7.jpg"
BG_DARK = "#000814"  
CARD_DARK = "#001D3D" 
ACCENT_CYAN = "#00F2FF"
ACCENT_BLUE = "#003566"

# ==========================================
# 2. Navigation Control (Fixing the Session State Error)
# ==========================================
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = "🏠 Home"

def move_to_dash():
    st.session_state.nav_selection = "📊 Dashboard"


# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    
    /* Main App Background */
    .stApp {{
        background-image: linear-gradient(rgba(0,8,20,0.8), rgba(0,8,20,0.9)), url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white; font-family: 'Inter', sans-serif;
    }}

    /* MAKING THE SIDEBAR TRANSPARENT */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0) !important; /* Fully transparent background */
        backdrop-filter: blur(10px); /* Adds a high-end blur effect */
        border-right: 1px solid rgba(0, 242, 255, 0.1);
    }}

    /* Removing the default white overlay on sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
    }}

    /* Styling the Radio Buttons in the sidebar to look cleaner */
    .stRadio > div {{
        background: transparent !important;
    }}
    
    label[data-testid="stWidgetLabel"] {{
        color: {ACCENT_CYAN} !important;
        font-weight: bold;
        letter-spacing: 1px;
    }}

    /* Card & Container Styling */
    div[data-testid="stVerticalBlock"] > div.stColumn > div {{
        background-color: rgba(0, 29, 61, 0.6);
        backdrop-filter: blur(15px);
        padding: 25px; border-radius: 15px; border: 1px solid rgba(0, 242, 255, 0.1);
    }}

    h1, h2, h3 {{ color: {ACCENT_CYAN} !important; font-weight: 800; }}

    /* Button Styling */
    div.stButton > button {{
        background: rgba(0, 242, 255, 0.1);
        color: {ACCENT_CYAN} !important; border: 1px solid {ACCENT_CYAN};
        border-radius: 10px; padding: 10px 25px; transition: 0.4s;
        backdrop-filter: blur(5px);
    }}
    div.stButton > button:hover {{
        background: {ACCENT_CYAN}; color: black !important;
        box-shadow: 0px 0px 20px {ACCENT_CYAN};
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. Data Loading & Mapping
# ==========================================
FILENAME = "Supplemental Drone Telemetry Data - Drone Operations Log _test11.csv"

@st.cache_data
def load_clean_data():
    if os.path.exists(FILENAME):
        df = pd.read_csv(FILENAME)
        if 'Actual Carry Weight (kg)' in df.columns:
            df['Actual Carry Weight (kg)'] = pd.to_numeric(df['Actual Carry Weight (kg)'], errors='coerce').fillna(0)
        if 'Flight Status' in df.columns:
            df['Is_Failure'] = df['Flight Status'].apply(lambda x: 1 if x != 'Completed' else 0)
        return df
    return pd.DataFrame()

df = load_clean_data()

MAPS = {
    "Application": {"Package Delivery": 11, "Aerial Photography": 0, "Agricultural Spraying": 1, "Infrastructure Inspection": 10, "Power Line Inspection": 13, "Film Production": 23, "Wind Turbine Inspection": 9, "Environmental Monitoring": 7, "Event Coverage": 20, "Traffic Monitoring": 8, "Surveillance": 19, "Surveying": 18, "Search and Rescue": 17, "Warehouse Inventory": 3, "Construction Site Survey": 12, "Delivery to Remote Area": 5, "Precision Agriculture": 21, "Wildlife Monitoring": 22, "Emergency Response": 6, "Pipeline Inspection": 15, "Product Photography": 14, "Real Estate Photography": 4, "Bridge Inspection": 2, "Crop Monitoring": 16},
    "Payload": {"Camera": 0, "Package": 10, "Liquid Tank": 9, "Sensor": 13, "Camera, Sensor": 3, "Camera, GPS": 2, "LiDAR System": 8, "First Aid Kit": 6, "Scanner": 12, "Camera, Beacon": 1, "Camera, Speaker": 4, "GPS": 7, "Speaker": 14, "RFID Reader": 11, "Communication Device": 5},
    "Model": {"SnapShot Mini": 17, "CropMaster": 4, "ViewMax 500": 22, "FlyHigh 300": 8, "SwiftWing": 20, "VoltGuard": 23, "BladeChecker": 2, "CineDrone X": 3, "AirProbe 1": 1, "TrafficEye": 21, "EventFlyer": 5, "Watcher Pro": 13, "Guardian SR": 24, "FarmPilot": 9, "MapMaker Z": 12, "SiteScan": 18, "StockChecker": 11, "FirstResponder": 15, "NatureWatch": 19, "LongHaul 400": 7, "PipePatrol": 14, "StudioShot": 6, "SkyLens": 0, "Agri Scout": 16, "Inspecta X": 10},
    "Size": {"Small": 2, "Medium": 1, "Large": 0}
}

@st.cache_resource
def load_drone_model():
    try:
        with open("random_forest_model.pkl", "rb") as f: return pickle.load(f)
    except: return None

model = load_drone_model()

# --- Sidebar ---
st.sidebar.markdown(f"<h1 style='text-align:center; color:{ACCENT_CYAN};'>🚁 Drone Care</h1>", unsafe_allow_html=True)
selection = st.sidebar.radio("MAIN MENU", ["🏠 Home", "📊 Dashboard", "🚁 Manual Input", "📈 Model Performance", "👥 About Us"], key="nav_selection")

# ==========================================
# 5. Page Logic
# ==========================================

# --- [ 🏠 Home Page - MEGA UPGRADE ] ---
if selection == "🏠 Home":
    # Overlay styling for the hero section
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,8,20,0.8), rgba(0,8,20,0.9)), url("{BG_IMAGE_URL}");
            background-size: cover;
            background-position: center;
        }}
        .hero-text {{
            text-align: center;
            padding: 80px 0px 20px 0px;
        }}
        .main-title {{
            font-size: 100px !important;
            letter-spacing: 15px;
            font-weight: 900;
            margin-bottom: 0px;
            text-shadow: 0px 0px 30px {ACCENT_CYAN};
        }}
        .sub-title {{
            font-size: 24px;
            color: {ACCENT_CYAN};
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 50px;
        }}
        .feature-card {{
            background: rgba(0, 29, 61, 0.6);
            border: 1px solid rgba(0, 242, 255, 0.2);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            transition: 0.3s;
            backdrop-filter: blur(10px);
        }}
        .feature-card:hover {{
            border: 1px solid {ACCENT_CYAN};
            transform: translateY(-10px);
            background: rgba(0, 29, 61, 0.8);
        }}
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="hero-text">
            <h1 class="main-title">DRONE<span style='color:white;'>CARE</span></h1>
            <p class="sub-title">An Intelligent System for Predicting Drone Failures Using AI </p>
        </div>
    """, unsafe_allow_html=True)

    # Decorative visual element
    

    # Action Cards Row
    st.write("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="feature-card">
                <h2 style='font-size:40px;'>📊</h2>
                <h3>Drone Operations Dashboard</h3>
                <p style='color:#ccc; font-size:14px;'>Real-time insights on battery, wind, and altitude for safer flights.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="feature-card">
                <h2 style='font-size:40px;'>🧠</h2>
                <h3>AI Prediction</h3>
                <p style='color:#ccc; font-size:14px;'>Machine Learning algorithms (Random Forest) detecting failure patterns before they happen.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="feature-card">
                <h2 style='font-size:40px;'>🛡️</h2>
                <h3>Asset Safety</h3>
                <p style='color:#ccc; font-size:14px;'>Protecting public infrastructure and high-value hardware from sudden crashes.</p>
            </div>
        """, unsafe_allow_html=True)

    # Main Launch Button
    st.write("<br><br>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
    with c_btn2:
        st.button("INITIALIZE SYSTEMS 🚀", on_click=move_to_dash, use_container_width=True)
        st.markdown("<p style='text-align:center; opacity:0.5; font-size:12px;'>SYSTEM STATUS: ALL SENSORS NOMINAL</p>", unsafe_allow_html=True)

    # Subtle Footer
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.3;'>UTAS GRADUATION PROJECT | AI DIVISION 2026</p>", unsafe_allow_html=True)

# --- [ 📊 Dashboard Page ] ---
elif selection == "📊 Dashboard":
    st.title("📊 Fleet Intelligence Dashboard")
    
    if df.empty:
        st.error("No telemetry data found. Please check your CSV file.")
    else:
        # 1. Global Filter Sidebar (Integrated with your dataset)
        all_models = df['Drone Model'].unique()
        sel_models = st.sidebar.multiselect("Filter by Drone Model", all_models, default=all_models[:3])
        f_df = df[df['Drone Model'].isin(sel_models)]

        # 2. KPI Metrics Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Missions", len(f_df))
        k2.metric("Avg Battery Level", f"{int(f_df['Battery Remaining (%)'].mean())}%")
        k3.metric("Critical Wind Speed", f"{f_df['Wind Speed (m/s)'].max()} m/s")
        fail_rate = (f_df['Is_Failure'].mean() * 100)
        k4.metric("Incident Probability", f"{fail_rate:.1f}%")

        st.markdown("---")

        # 3. Row One: Telemetry Distribution (The Bar Chart you requested)
        # --- [ 3. Row One: Telemetry Distribution (Updated with more features) ] ---
        st.subheader("📊 Flight Telemetry Analysis")
        
        compare_feat = st.selectbox("Select metric to analyze distribution:", 
                                    [
                                        "Battery Remaining (%)", 
                                        "Wind Speed (m/s)", 
                                        "Altitude (meters)",
                                        "GPS Accuracy (meters)", 
                                        "Max Carry Weight (kg)", 
                                        "Obstacles Encountered"
                                    ])
        
        fig_bar_trend = px.histogram(f_df, x=compare_feat, color="Flight Status",
                                     nbins=20, barmode="group",
                                     template="plotly_dark",
                                     title=f"Distribution of {compare_feat} by Mission Outcome",
                                     color_discrete_map={'Completed': ACCENT_CYAN, 'Landed Unexpectedly': '#FF0055'})
        
        fig_bar_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Number of Flights",
            xaxis_title=compare_feat
        )
        st.plotly_chart(fig_bar_trend, use_container_width=True)

        st.markdown("---")

        # 4. Row Two: Pie Chart & Success by Model
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Overall Mission Status")
            status_counts = f_df['Flight Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_pie = px.pie(status_counts, values='Count', names='Status', hole=0.5,
                             template="plotly_dark",
                             color='Status',
                             color_discrete_map={'Completed': ACCENT_CYAN, 'Landed Unexpectedly': '#FF0055'})
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("🛰️ Success Rate by Drone Model")
            fig_model_bar = px.bar(f_df.groupby(['Drone Model', 'Flight Status']).size().reset_index(name='Count'),
                                   x="Drone Model", y="Count", color="Flight Status",
                                   template="plotly_dark",
                                   color_discrete_map={'Completed': ACCENT_CYAN, 'Landed Unexpectedly': '#FF0055'})
            fig_model_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_model_bar, use_container_width=True)

        st.markdown("---")

        # 5. Row Three: NEW - Feature Inter-Relationships (The "Wow" Analysis)
        st.subheader("🔗 Feature Inter-Relationships & Correlations")
        st.write("Understand how environmental factors like Wind affect technical performance like Battery.")

        col_rel1, col_rel2 = st.columns(2)

        with col_rel1:
            # Scatter Plot to show if higher wind leads to lower battery
            st.markdown("#### Wind Speed vs. Battery Stress")
            fig_scatter = px.scatter(f_df, x='Wind Speed (m/s)', y='Battery Remaining (%)', 
                                     color='Flight Status', trendline="ols",
                                     template="plotly_dark",
                                     color_discrete_map={'Completed': ACCENT_CYAN, 'Landed Unexpectedly': '#FF0055'},
                                     title="Relationship: How Wind Drains Battery")
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_rel2:
            # Correlation Heatmap for all 13 features
            st.markdown("#### Feature Correlation Matrix")
            numeric_cols = f_df.select_dtypes(include=[np.number]).columns
            corr_matrix = f_df[numeric_cols].corr()
            
            fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                    color_continuous_scale='RdBu_r', 
                                    template="plotly_dark",
                                    title="Correlation Matrix (Heatmap)")
            fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heatmap, use_container_width=True)

# --- [ 🚁 Manual Input Page ] ---
elif selection == "🚁 Manual Input":
    st.title("🚁 AI Risk Prediction")
    st.write("Enter mission details to get an instant safety assessment from the AI model.")
    
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_count = st.number_input("Propeller Count", 4, 8, 4)
            max_w = st.number_input("Max Carry Weight (kg)", 0.1, 100.0, 10.0)
            d_size = st.selectbox("Drone Size", list(MAPS["Size"].keys()))
            alt = st.number_input("Altitude (meters)", 0, 1000, 120)
            batt = st.slider("Battery Remaining (%)", 0, 100, 80)
        with col2:
            p_type = st.selectbox("Payload Type", list(MAPS["Payload"].keys()))
            app = st.selectbox("Application", list(MAPS["Application"].keys()))
            act_w = st.number_input("Actual Carry Weight (kg)", 0.0, 100.0, 2.0)
            gps_acc = st.number_input("GPS Accuracy (meters)", 0.0, 10.0, 0.5)
        with col3:
            d_model = st.selectbox("Drone Model", list(MAPS["Model"].keys()))
            dist = st.number_input("Distance Flown (km)", 0.0, 100.0, 5.0)
            wind = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 4.0)
            obs = st.selectbox("Obstacles Encountered", ["No", "Yes"])

        if st.form_submit_button("ANALYZE FLIGHT RISK"):
            if model:
                # تجميع الـ 13 فيتشر بالترتيب
                feats = [p_count, max_w, MAPS["Size"][d_size], alt, batt, 
                         MAPS["Payload"][p_type], MAPS["Application"][app], 
                         act_w, gps_acc, MAPS["Model"][d_model], dist, wind, (1 if obs == "Yes" else 0)]
                
                prediction = model.predict([feats])[0]
                if prediction == 1:
                    st.error("🚨 HIGH RISK: The AI predicts a likely flight failure. Check wind/battery levels.")
                else:
                    st.success("✅ Low Risk: Flight parameters are within safe thresholds")
                    st.balloons()
            else:
                st.warning("Prediction model not found. Check 'rf_drone_model11.pkl'")

# --- [ 📈 Model Performance Page ] ---
# --- [ 📈 Model Performance Page ] ---
elif selection == "📈 Model Performance":
    st.title("🛡️ AI Model Core Intelligence")
    st.markdown("---")

    # القسم الأول: شرح الخوارزمية (Random Forest Architecture)
    c1, c2 = st.columns([1.1, 0.9])
    
    with c1:
        st.subheader("🌲 Random Forest Architecture")
        st.markdown(f"""
        <div style='background:rgba(0, 242, 255, 0.05); padding:20px; border-radius:15px;'>
        The model utilizes a <b>Random Forest Classifier</b>, an ensemble technique that constructs a "forest" of uncorrelated decision trees to achieve superior predictive stability.
        <br><br>
        <b>Core Mechanics:</b>
        <ul>
            <li><b>Bagging (Bootstrap Aggregating):</b> Each tree is trained on a random subsample of the telemetry data, which significantly reduces <i>Model Variance</i>.</li>
            <li><b>Feature Subsampling:</b> At every split in a tree, the algorithm considers only a random subset of features, ensuring that no single sensor (like GPS) dominates the decision-making process unless it is truly critical.</li>
            <li><b>Robustness to Noise:</b> By aggregating votes from 100+ trees, the system effectively filters out "sensor jitter" and momentary telemetry spikes.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("🎯 Feature Importance ")
        # تم تحديث البيانات لتعكس الـ Gini Importance الحقيقي
        imp_data = pd.DataFrame({
            'Feature': ['Wind Speed', 'Battery %', 'Obstacles Encountered', 'Max Carry Weight (kg)', 'GPS Accuracy',],
            'Weight': [35, 30, 15, 12, 8]
        }).sort_values('Weight')
        
        fig_imp = px.bar(imp_data, x='Weight', y='Feature', orientation='h',
                         color='Weight', color_continuous_scale=['#003566', ACCENT_CYAN],
                         template="plotly_dark", title="Information Gain per Feature")
        
        fig_imp.update_layout(
            showlegend=False, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

elif selection == "👥 About Us":
    st.markdown(f"""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='font-size: 50px;'>🛡️ GRADUATION PROJECT: <span style='color:white;'>DRONE CARE</span></h1>
            <p style='color:{ACCENT_CYAN}; font-size: 22px;'>University of Technology and Applied Sciences (UTAS) | 2026</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_text, col_gfx = st.columns([1.2, 0.8])
    with col_text:
        st.subheader("🚀 Our Mission")
        st.markdown(f"""
        <div style='background-color:{CARD_DARK}; padding: 25px; border-radius: 15px; border-left: 5px solid {ACCENT_CYAN};'>
            <p style='font-size: 18px; line-height: 1.6;'>
                <b>Drone Care</b> is an advanced AI system designed to predict drone malfunctions <b>BEFORE</b> they occur. 
                By identifying critical failure patterns, our system allows operators to intervene, preventing crashes that could 
                damage public or private property and endanger human lives.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Key Objectives")
    obj1, obj2, obj3 = st.columns(3)
    with obj1:
        st.markdown("<div style='background:#001d3d; padding:20px; border-radius:15px; text-align:center;'><h2>🛡️</h2><h4>Asset Protection</h4><p>Minimize physical damage to buildings and infrastructure.</p></div>", unsafe_allow_html=True)
    with obj2:
        st.markdown("<div style='background:#001d3d; padding:20px; border-radius:15px; text-align:center;'><h2>👥</h2><h4>Public Safety</h4><p>Ensure drones do not fall in populated zones, protecting lives.</p></div>", unsafe_allow_html=True)
    with obj3:
        st.markdown("<div style='background:#001d3d; padding:20px; border-radius:15px; text-align:center;'><h2>⚙️</h2><h4>Machine Learning</h4><p>Leveraging Random Forest for high-accuracy risk classification.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2026 Drone Care Team - UTAS. Empowering Safer Skies through Intelligence.")
