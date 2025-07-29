import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from datetime import datetime

@st.cache_data
def load_full_df():
    # Replace with your actual data loading, e.g.:
    return pd.read_csv("Flat prices(mld project).csv")

full_df = load_full_df()

st.subheader(f"Market Insights for {st.session_state.selected_town}")

if 'selected_town' not in st.session_state:
    st.session_state.selected_town = 'ANG MO KIO'  # default

st.subheader(f"Market Insights for {st.session_state.selected_town}")


# Inputs for analysis
floor_area = st.number_input(
    "Floor Area for Analysis (sqm)",
    min_value=20.0, max_value=300.0, value=90.0, step=1.0,
    key='insight_floor'
)
year = st.number_input(
    "Analysis Year",
    min_value=1990, max_value=datetime.now().year + 1, value=datetime.now().year, step=1,
    key='insight_year'
)
month = st.number_input(
    "Analysis Month",
    min_value=1, max_value=12, value=6, step=1,
    key='insight_month'
)
budget = st.number_input(
    "Your Budget (SGD)",
    min_value=100000, max_value=2000000, value=500000, step=10000,
    key='insight_budget'
)

flat_types = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM',
    'EXECUTIVE', 'MULTI-GENERATION'
]
storey_ranges = [
    '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
    '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
    '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
]

selected_flat_type = st.selectbox(
    "Select Flat Type for Trend Analysis", flat_types, index=2
)
selected_storey_range = st.selectbox(
    "Select Storey Range for Trend Analysis", storey_ranges, index=3
)

show_charts = st.checkbox("Show Charts", value=True)

# Main prediction loop
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
            st.error(f"Prediction error: {e}")

results_df = pd.DataFrame(results)

st.metric(
    "Average Price in Area",
    format_price(results_df['Price'].mean()),
    help="Average across all flat types and storey ranges"
)

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

# --- DYNAMIC CHARTS ---
import matplotlib.pyplot as plt
import seaborn as sns

if show_charts:
    col1, col2 = st.columns(2)

    # Monthly trend for selected flat + storey
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

        try:
            pred_price = model.predict(input_df.values)[0]
            monthly_prices.append(pred_price)
        except Exception:
            monthly_prices.append(None)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 3), dpi=150)
        ax1.plot(months, monthly_prices, marker='o')
        ax1.set_title(f"{selected_flat_type} Price Trend ({selected_storey_range}, {year})")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Predicted Price (SGD)")
        ax1.grid(True)
        st.pyplot(fig1)

    # Price distribution across all flat types
    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 3))
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
