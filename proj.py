import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from datetime import datetime

import matplotlib.pyplot as plt

@st.cache_data
def load_full_df():
 
    return pd.read_csv("Flat prices(mld project).csv")

full_df = load_full_df()


if 'year' not in full_df.columns:
    
    if 'month' in full_df.columns:
        
        full_df['month'] = pd.to_datetime(full_df['month'], errors='coerce')
        full_df['year'] = full_df['month'].dt.year
        full_df['month_num'] = full_df['month'].dt.month
    elif 'resale_date' in full_df.columns:
        full_df['resale_date'] = pd.to_datetime(full_df['resale_date'], errors='coerce')
        full_df['year'] = full_df['resale_date'].dt.year
        full_df['month_num'] = full_df['resale_date'].dt.month
    else:
        st.error("Your dataset needs a 'year' or date column to generate trends.")
        st.stop()


pdk.settings.mapbox_api_key = "YOUR_MAPBOX_PUBLIC_KEY"

st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


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


town_data = load_town_data()
model = load_model()


EXPECTED_COLUMNS = [
    'floor_area_sqm', 'year', 'month_num',
    *['town_' + t for t in town_data['Town']],
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
    'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
    *['storey_range_' + s for s in [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', 
        '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24',
        '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
        '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
    ]]
]


if 'view_state' not in st.session_state:
    st.session_state.view_state = {
        'latitude': 1.3521,
        'longitude': 103.8198,
        'zoom': 11.5,
        'pitch': 0
    }

if 'selected_town' not in st.session_state:
    st.session_state.selected_town = 'ANG MO KIO'

def format_price(price):
    """Format price as SGD currency."""
    return f"${price:,.0f}"


st.title("🏠 Singapore HDB Resale Price Predictor")
st.markdown("""
    <div class='header-text'>
        Predict resale prices for HDB flats across Singapore using our machine learning model.
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏡 Price Prediction", 
    "🗺️ Singapore Map", 
    "📊 Market Insights",
    "⚖️ Compare Towns"   
])

def update_map_center(town):
    """Update map view based on selected town."""
    selected = town_data[town_data['Town'] == town].iloc[0]
    st.session_state.view_state.update({
        'latitude': selected['Latitude'],
        'longitude': selected['Longitude'],
        'zoom': 13
    })
    st.session_state.selected_town = town




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

    if submitted:
        valid = True
        if floor_area <= 0:
            st.error("Floor Area should not be lesser than 0.")
            valid = False
        if year < 1990:
            st.error("Year should not be lesser than 1990.")
            valid = False
        if month < 1 or month > 12:
            st.error("Month should be between 1 and 12.")
            valid = False

        if valid:
            update_map_center(town)

        with col2:
            st.subheader("Prediction Results")
            if submitted and valid:
                with st.expander("Why this price?"):
                    st.write("""
                        - Larger floor area increases price.
                        - Town affects price due to location desirability.
                        - Flat type influences value (e.g., EXECUTIVE usually more expensive).
                        - Storey range can affect views and demand.
                        - Market conditions at the resale date impact price.
                    """)
                    
                    feature_importances = pd.Series(model.feature_importances_, index=EXPECTED_COLUMNS)

               
                    top_features = feature_importances.nlargest(15)

                    fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
                    top_features.sort_values().plot(kind='barh', color='green', ax=ax_imp)
                    ax_imp.set_title("Top 15 Feature Importances - Random Forest")
                    st.pyplot(fig_imp)

                    st.caption("""
                    - **Higher bars** = feature has more influence on price predictions.  
                    - Example: `town_QUEENSTOWN` high → location strongly impacts price.  
                    """)


                with st.spinner("Calculating estimate..."):
                   
                    input_df = pd.DataFrame({
                        'floor_area_sqm': [floor_area],
                        'year': [year],
                        'month_num': [month],
                        'town': [town],
                        'flat_type': [flat_type],
                        'storey_range': [storey_range]
                    })

                    
                    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])

                   
                    for col in EXPECTED_COLUMNS:
                        input_df[col] = input_df.get(col, 0)
                    input_df = input_df[EXPECTED_COLUMNS]

                   
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
                        st.stop()

                    

                    years = list(range(2017, year + 1))
                    pred_rows = []

                    for y in years:
                        yearly_df = pd.DataFrame({
                            'floor_area_sqm': [floor_area],
                            'year': [y],
                            'month_num': [6],
                            'town': [town],
                            'flat_type': [flat_type],
                            'storey_range': [storey_range]
                        })
                        yearly_df = pd.get_dummies(yearly_df, columns=['town', 'flat_type', 'storey_range'])
                        for col in EXPECTED_COLUMNS:
                            yearly_df[col] = yearly_df.get(col, 0)
                        yearly_df = yearly_df[EXPECTED_COLUMNS]

                        pred_price = model.predict(yearly_df.values)[0]
                        pred_rows.append({'Year': y, 'Predicted Price': pred_price})

                    df_pred = pd.DataFrame(pred_rows)

                   
                    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                    ax.plot(df_pred['Year'], df_pred['Predicted Price'], marker='o', color='blue')
                    ax.set_title(f"Predicted Price Trend in {town} ({flat_type}, {storey_range})")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Predicted Price (SGD)")
                    ax.grid(True)
                    st.pyplot(fig)

                   


                
with tab2:
    st.subheader("HDB Towns in Singapore")

    with st.expander("Map Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            zoom = st.slider("Zoom Level", 9.0, 15.0, float(st.session_state.view_state['zoom']), 0.5)
        with col2:
            pitch = st.slider("Map Tilt", 0, 60, st.session_state.view_state['pitch'], 5)
        with col3:
            point_size = st.slider("Marker Size", 50, 500, 200, 50)

        filter_flat_types = st.multiselect(
            "Select Flat Types",
            options=full_df['flat_type'].unique().tolist(),
            default=full_df['flat_type'].unique().tolist()
        )


    st.session_state.view_state.update({'zoom': zoom, 'pitch': pitch})
    map_placeholder = st.empty()


    with map_placeholder.container():
        st.info("⏳ Loading map data...")

    filtered = full_df[full_df['flat_type'].isin(filter_flat_types)]

    if filtered.empty:
   
        map_df = town_data.copy()
        map_df['Tooltip'] = "None set or flat room not found"
    else:
        
        avg_price_by_town_flat = (
            filtered.groupby(['town', 'flat_type'])['resale_price']
            .mean().reset_index()
        )

      
        tooltip_series = (
            avg_price_by_town_flat
            .groupby('town')
            .apply(lambda df: '<br>'.join(
                f"{ft}: ${p:,.0f}" for ft, p in zip(df['flat_type'], df['resale_price'])
            ))
        )


        tooltip_texts = pd.DataFrame({
            'town': tooltip_series.index,
            'Tooltip': tooltip_series.values
        })

        map_df = town_data.merge(tooltip_texts, left_on='Town', right_on='town', how='left')
        map_df = map_df[['Town', 'Latitude', 'Longitude', 'Tooltip']]
        map_df['Tooltip'] = map_df['Tooltip'].fillna("No Data") 

    with map_placeholder.container():

        map_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[Longitude, Latitude]',
            get_radius=point_size,
            get_fill_color='[255, 140, 0, 180]',
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=zoom,
            pitch=pitch
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[map_layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{Town}</b><br>{Tooltip}"}
            ),
            use_container_width=True,
            height=800
        )

with tab3:
    st.subheader(f"Market Insights for {st.session_state.selected_town}")

   
    floor_area = st.number_input(
        "Floor Area for Analysis (sqm)", 20.0, 300.0, 90.0, step=1.0, key='insight_floor'
    )
    year = st.number_input(
        "Analysis Year", 1990, datetime.now().year + 1, datetime.now().year, step=1, key='insight_year'
    )
    month = st.number_input(
        "Analysis Month", 1, 12, 6, step=1, key='insight_month'
    )
    budget = st.number_input(
        "Your Budget (SGD)", 100000, 2000000, 500000, step=10000, key='insight_budget'
    )

    flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    storey_ranges = [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', 
        '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24',
        '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
        '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
    ]

    selected_flat_type = st.selectbox("Select Flat Type for Trend Analysis", flat_types, index=2)
    selected_storey_range = st.selectbox("Select Storey Range for Trend Analysis", storey_ranges, index=3)
    show_charts = st.checkbox("Show Charts", value=True)

  
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

            pred_price = model.predict(input_df.values)[0]
            results.append({
                'Flat Type': f_type,
                'Storey Range': storey,
                'Price': pred_price,
                'Within Budget': pred_price <= budget
            })

    results_df = pd.DataFrame(results)

    st.metric("Average Price in Area", format_price(results_df['Price'].mean()), help="Average across all flat types and storey ranges")

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


    if show_charts:
        import matplotlib.pyplot as plt
        import seaborn as sns

      
        months = pd.date_range(start=f"{year}-01", periods=12, freq='M')
        monthly_prices = []
        for m in range(1, 13):
            input_df = pd.DataFrame({
                'floor_area_sqm': [floor_area],
                'year': [year],
                'month_num': [m],
                'town': [st.session_state.selected_town],
                'flat_type': [selected_flat_type],
                'storey_range': [selected_storey_range]
            })
            input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
            for col in EXPECTED_COLUMNS:
                input_df[col] = input_df.get(col, 0)
            input_df = input_df[EXPECTED_COLUMNS]

            monthly_prices.append(model.predict(input_df.values)[0])

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
            ax1.plot(months, monthly_prices, marker='o')
            ax1.set_title(f"{selected_flat_type} Price Trend ({selected_storey_range}, {year})")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Predicted Price (SGD)")
            ax1.grid(True)
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=130)
            sns.histplot(
                data=results_df,
                x='Price',
                hue='Flat Type',
                multiple='stack',
                kde=True,
                palette='Set2',
                ax=ax2
            )
            ax2.set_title("Price Distribution by Flat Type")
            ax2.set_xlabel("Price (SGD)")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

with tab4:
    st.subheader("Compare Two Towns/Flats")


    st.markdown("### Comparison Inputs")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        town1 = st.selectbox("Select Town 1", sorted(town_data['Town'].unique()), key='compare_town1')
        flat1 = st.selectbox("Flat Type 1", flat_types, key='compare_flat1')
        storey1 = st.selectbox("Storey Range 1", storey_ranges, key='compare_storey1')

    with input_col2:
        town2 = st.selectbox("Select Town 2", sorted(town_data['Town'].unique()), key='compare_town2')
        flat2 = st.selectbox("Flat Type 2", flat_types, key='compare_flat2')
        storey2 = st.selectbox("Storey Range 2", storey_ranges, key='compare_storey2')

    compare_area = st.number_input("Floor Area (sqm) for Comparison", 20.0, 300.0, 90.0, 1.0)

    if st.button("Compare Prices"):
        def predict_price(town, flat, storey, area, year=None, month=None):
            year = year or datetime.now().year
            month = month or datetime.now().month
            df = pd.DataFrame({
                'floor_area_sqm': [area],
                'year': [year],
                'month_num': [month],
                'town': [town],
                'flat_type': [flat],
                'storey_range': [storey]
            })
            df = pd.get_dummies(df, columns=['town','flat_type','storey_range'])
            for col in EXPECTED_COLUMNS:
                df[col] = df.get(col, 0)
            df = df[EXPECTED_COLUMNS]
            return model.predict(df.values)[0]

     
        price1 = predict_price(town1, flat1, storey1, compare_area)
        price2 = predict_price(town2, flat2, storey2, compare_area)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Estimated Price Town 1", format_price(price1))
        metric_col2.metric("Estimated Price Town 2", format_price(price2))

        diff = price1 - price2
        st.write(f"💡 **Price Difference:** {format_price(abs(diff))} ({'Town 1 higher' if diff>0 else 'Town 2 higher'})")

        import matplotlib.pyplot as plt

        years = list(range(2017, datetime.now().year + 1))
        trend1, trend2 = [], []

        for y in years:
            trend1.append(predict_price(town1, flat1, storey1, compare_area, year=y, month=6))
            trend2.append(predict_price(town2, flat2, storey2, compare_area, year=y, month=6))

        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100) 
        ax.plot(years, trend1, marker='o', color='blue', label=f"{town1} ({flat1})")
        ax.plot(years, trend2, marker='o', color='green', label=f"{town2} ({flat2})")
        ax.set_title("Price Trend Comparison")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted Price (SGD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

st.markdown("---")
st.markdown("""
    <div class='footer'>
        Note: Predictions are based on historical data and machine learning models. 
        Actual prices may vary depending on market conditions and property specifics.
        <br>Last updated: {}
    </div>
""".format(datetime.now().strftime("%d %B %Y")), unsafe_allow_html=True)

