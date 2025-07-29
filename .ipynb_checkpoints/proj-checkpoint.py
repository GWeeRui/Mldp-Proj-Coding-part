import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING ---
def apply_custom_styles():
    st.markdown("""
        <style>
            .main {background-color: #f8f9fa;}
            .stButton>button {border-radius: 4px; padding: 8px 16px;}
            .stSelectbox, .stNumberInput {border-radius: 4px;}
            .stTabs [data-baseweb="tab-list"] {gap: 4px;}
            .stTabs [data-baseweb="tab"] {border-radius: 4px !important;}
            .metric-value {font-size: 1.5rem !important;}
            .header-text {color: #1a3d7c;}
            .footer {font-size: 0.8rem; color: #666;}
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# --- DATA LOADING ---
@st.cache_data
def load_town_data():
    """Load geographical data for HDB towns."""
    return pd.DataFrame({
        'Town': [
            'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
            'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA',
            'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG',
            'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
        ],
        'Latitude': [
            1.3691, 1.3244, 1.3509, 1.3496, 1.2840,
            1.3786, 1.3294, 1.2897, 1.3854, 1.3151,
            1.3180, 1.3711, 1.3331, 1.3391, 1.3194,
            1.3039, 1.3736, 1.4043, 1.2945, 1.4432,
            1.3911, 1.3500, 1.3496, 1.3354, 1.4360, 1.4296
        ],
        'Longitude': [
            103.8454, 103.9310, 103.8485, 103.7490, 103.8200,
            103.7634, 103.8021, 103.8519, 103.7441, 103.7654,
            103.8845, 103.8930, 103.7421, 103.7066, 103.8622,
            103.9120, 103.9497, 103.9020, 103.8060, 103.8201,
            103.8958, 103.8739, 103.9456, 103.8501, 103.7865, 103.8355
        ]
    })

@st.cache_resource
def load_model():
    """Load the trained prediction model."""
    try:
        return joblib.load("resale_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'resale_model.pkl' exists.")
        st.stop()

# Initialize data and model
town_data = load_town_data()
model = load_model()

# --- MODEL CONFIGURATION ---
EXPECTED_COLUMNS = [
    'floor_area_sqm', 'year', 'month_num',
    *['town_' + t for t in town_data['Town']],
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
    'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
    *['storey_range_' + s for s in [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
        '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
        '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
    ]]
]

# --- SESSION STATE MANAGEMENT ---
if 'view_state' not in st.session_state:
    st.session_state.view_state = {
        'latitude': 1.3521,
        'longitude': 103.8198,
        'zoom': 11.5,
        'pitch': 0
    }

if 'selected_town' not in st.session_state:
    st.session_state.selected_town = 'ANG MO KIO'

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def update_map_center(town):
    """Update map view based on selected town."""
    selected = town_data[town_data['Town'] == town].iloc[0]
    st.session_state.view_state.update({
        'latitude': selected['Latitude'],
        'longitude': selected['Longitude'],
        'zoom': 13
    })
    st.session_state.selected_town = town

def format_price(price):
    """Format price as SGD currency."""
    return f"${price:,.0f}"

# --- MAIN APP ---
st.title("🏠 Singapore HDB Resale Price Predictor")
st.markdown("""
    <div class='header-text'>
        Predict resale prices for HDB flats across Singapore using our machine learning model.
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🏡 Price Prediction", 
    "🗺️ Singapore Map", 
    "📊 Market Insights"
])

# --- PRICE PREDICTION TAB ---
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Property Details")
        with st.form("prediction_form"):
            floor_area = st.number_input(
                "Floor Area (sqm)",
                min_value=20.0,
                max_value=300.0,
                value=90.0,
                step=1.0
            )
            year = st.number_input(
                "Year of Resale",
                min_value=1990,
                max_value=datetime.now().year + 1,
                value=datetime.now().year,
                step=1
            )
            month = st.number_input(
                "Month of Resale",
                min_value=1,
                max_value=12,
                value=6,
                step=1
            )
            town = st.selectbox(
                "Town",
                sorted(town_data['Town'].unique()),
                key='town_selectbox'
            )
            flat_type = st.selectbox(
                "Flat Type",
                ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', 
                 '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
            )
            storey_range = st.selectbox(
                "Storey Range",
                ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', 
                 '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24',
                 '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
                 '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
            )
            
            submitted = st.form_submit_button("Predict Price", type="primary")

    with col2:
        st.subheader("Prediction Results")
        if submitted:
            # Update map center when form is submitted
            update_map_center(town)
            
            with st.spinner("Calculating estimate..."):
                # Prepare input data
                input_df = pd.DataFrame({
                    'floor_area_sqm': [floor_area],
                    'year': [year],
                    'month_num': [month],
                    'town': [town],
                    'flat_type': [flat_type],
                    'storey_range': [storey_range]
                })
                
                # One-hot encode categorical variables
                input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
                
                # Ensure all expected columns are present
                for col in EXPECTED_COLUMNS:
                    input_df[col] = input_df.get(col, 0)
                
                input_df = input_df[EXPECTED_COLUMNS]
                
                # Make prediction
                try:
                    prediction = model.predict(input_df.values)[0]
                    st.metric(
                        label="Estimated Resale Price",
                        value=format_price(prediction),
                        help=f"Prediction for {flat_type} flat in {town}"
                    )
                    st.markdown(f"""
                        <div style='margin-top: 20px; padding: 12px; background: #f0f2f6; border-radius: 4px; color: black;'>
                            <small>For a <strong>{flat_type.lower()}</strong> flat in <strong>{town}</strong><br>
                            {storey_range} floor range • {floor_area} sqm<br>
                            {datetime(year, month, 1).strftime('%B %Y')}</small>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# --- MAP TAB ---
with tab2:
    st.subheader("HDB Towns in Singapore")
    
    # Map controls
    with st.expander("Map Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            zoom = st.slider(
                "Zoom Level",
                min_value=9.0,
                max_value=15.0,
                value=float(st.session_state.view_state['zoom']),
                step=0.5
            )
        with col2:
            pitch = st.slider(
                "Map Tilt",
                min_value=0,
                max_value=60,
                value=st.session_state.view_state['pitch'],
                step=5
            )
        with col3:
            point_size = st.slider(
                "Marker Size",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )
            if st.button("Toggle Map Theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
    
    st.session_state.view_state.update({'zoom': zoom, 'pitch': pitch})
    
    # Prepare map data
    selected_town = town_data[town_data['Town'] == st.session_state.selected_town].iloc[0]
    highlight_data = pd.DataFrame({
        'Latitude': [selected_town['Latitude']],
        'Longitude': [selected_town['Longitude']],
        'Town': [st.session_state.selected_town]
    })
    
    # Create map layers
    highlight_layer = pdk.Layer(
        "ScatterplotLayer",
        data=highlight_data,
        get_position='[Longitude, Latitude]',
        get_radius=point_size,
        get_fill_color='[0, 100, 255, 200]',
        pickable=True
    )
    
    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=town_data,
        get_position='[Longitude, Latitude]',
        get_radius=point_size,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True,
        auto_highlight=True
    )
    
    # Render map
    map_style = "mapbox://styles/mapbox/dark-v10" if st.session_state.dark_mode else "mapbox://styles/mapbox/light-v9"
    
    st.pydeck_chart(
        pdk.Deck(
            map_style=map_style,
            layers=[base_layer, highlight_layer],
            initial_view_state=pdk.ViewState(**st.session_state.view_state),
            tooltip={"html": "<b>{Town}</b><br>Lat: {Latitude:.4f}, Lon: {Longitude:.4f}"}
        ),
        use_container_width=True
    )

# --- INSIGHTS TAB ---
with tab3:
    st.subheader(f"Market Insights for {st.session_state.selected_town}")
    
    # Get parameters from the prediction tab or use defaults
    floor_area = st.number_input(
        "Floor Area for Analysis (sqm)",
        min_value=20.0,
        max_value=300.0,
        value=90.0,
        step=1.0,
        key='insight_floor'
    )
    
    year = st.number_input(
        "Analysis Year",
        min_value=1990,
        max_value=datetime.now().year + 1,
        value=datetime.now().year,
        step=1,
        key='insight_year'
    )
    
    month = st.number_input(
        "Analysis Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        key='insight_month'
    )
    
    budget = st.number_input(
        "Your Budget (SGD)",
        min_value=100000,
        max_value=2000000,
        value=500000,
        step=10000,
        key='insight_budget'
    )
    
    if st.button("Generate Insights", key='insight_btn'):
        with st.spinner("Analyzing market data..."):
            # Generate predictions for all combinations
            flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
            storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
                           '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
                           '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
            
            results = []
            for f_type in flat_types:
                for storey in storey_ranges:
                    input_df = pd.DataFrame({
                        'floor_area_sqm': [floor_area],
                        'year': [year],
                        'month_num': [month],
                        'town': [st.session_state.selected_town],
                        'flat_type': [f_type],
                        'storey_range': [storey]
                    })
                    
                    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
                    for col in EXPECTED_COLUMNS:
                        input_df[col] = input_df.get(col, 0)
                    input_df = input_df[EXPECTED_COLUMNS]
                    
                    try:
                        pred_price = model.predict(input_df.values)[0]
                        results.append({
                            'Flat Type': f_type,
                            'Storey Range': storey,
                            'Price': pred_price,
                            'Within Budget': pred_price <= budget
                        })
                    except Exception as e:
                        st.error(f"Error predicting {f_type} at {storey}: {str(e)}")
            
            if results:
                results_df = pd.DataFrame(results)
                avg_price = results_df['Price'].mean()
                
                st.metric(
                    "Average Price in Area",
                    format_price(avg_price),
                    help="Average across all flat types and storey ranges"
                )
                
                # Filter affordable options
                affordable = results_df[results_df['Within Budget']]
                
                if not affordable.empty:
                    st.success(f"Found {len(affordable)} options within your budget")
                    st.dataframe(
                        affordable[['Flat Type', 'Storey Range', 'Price']]
                        .sort_values('Price')
                        .style.format({'Price': format_price}),
                        use_container_width=True
                    )
                else:
                    st.warning("No options found within your budget")
                    st.info("Here are the 5 most affordable options:")
                    st.dataframe(
                        results_df[['Flat Type', 'Storey Range', 'Price']]
                        .nsmallest(5, 'Price')
                        .style.format({'Price': format_price}),
                        use_container_width=True
                    )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div class='footer'>
        Note: Predictions are based on historical data and machine learning models. 
        Actual prices may vary depending on market conditions and property specifics.
        <br>Last updated: {}
    </div>
""".format(datetime.now().strftime("%d %B %Y")), unsafe_allow_html=True)


# Streamlit input validation
area = st.number_input("Floor Area (sqm)", min_value=1.0, step=1.0)

if area <= 0:
    st.error("Floor area must be greater than 0.")
