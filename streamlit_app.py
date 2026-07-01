"""Road-safety dashboard, reads only the Gold layer (data/gold/*.parquet).

Deployed via Streamlit Community Cloud, linked to the GitHub repo.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Road Safety, France 2024", layout="wide")

st.image(str(Path(__file__).parent / "medallion_architecture.png"))
st.caption("Bronze → Silver → Gold → this dashboard.")

GOLD = Path(__file__).parent / "data" / "gold"


@st.cache_data
def load_gold():
    fact = pd.read_parquet(GOLD / "fact_accidents.parquet")
    dim_location = pd.read_parquet(GOLD / "dim_location.parquet")
    dim_time = pd.read_parquet(GOLD / "dim_time.parquet")
    dim_conditions = pd.read_parquet(GOLD / "dim_conditions.parquet")
    dim_vehicle = pd.read_parquet(GOLD / "dim_vehicle.parquet")
    return fact, dim_location, dim_time, dim_conditions, dim_vehicle


fact, dim_location, dim_time, dim_conditions, dim_vehicle = load_gold()

# fact_accidents is person-grain, dedup on location_id first so KPIs count accidents, not people.
accidents_view = (
    fact.drop_duplicates("location_id")
    .merge(dim_location, on="location_id", how="left")
    .merge(dim_time, on="time_id", how="left")
    .merge(dim_conditions, on="condition_id", how="left")
)

lum_labels = {1: "Daylight", 2: "Dusk/dawn", 3: "Night, no lighting",
              4: "Night, lighting off", 5: "Night, lighting on"}
col_labels = {1: "Head-on", 2: "Rear-end", 3: "Side", 4: "Chain",
              5: "Multiple collisions", 6: "Other", 7: "No collision"}
severity_labels = {"mortel": "Fatal", "hospitalise": "Hospitalised", "blesse_leger": "Light injury"}

st.title("Road safety, bodily-injury accidents, France 2024")
st.caption("Source: ONISR / data.gouv.fr (BAAC file). Gold layer data, "
           "see notebook.ipynb for methodology.")

st.sidebar.header("Filters")

years = sorted(dim_time["year"].dropna().unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

deps = sorted(accidents_view["dep"].dropna().unique())
selected_deps = st.sidebar.multiselect("Department (INSEE code)", deps, default=[])

severity_options = sorted(accidents_view["accident_severity_class"].dropna().unique())
selected_severity = st.sidebar.multiselect(
    "Accident severity (worst outcome)", severity_options,
    default=severity_options, format_func=lambda s: severity_labels.get(s, s),
)

filtered = accidents_view[accidents_view["year"].isin(selected_years)]
if selected_deps:
    filtered = filtered[filtered["dep"].isin(selected_deps)]
filtered = filtered[filtered["accident_severity_class"].isin(selected_severity)]

if filtered.empty:
    st.warning("No accidents match the selected filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accidents", f"{len(filtered):,}")
col2.metric("People involved", f"{filtered['n_people'].sum():,}")
col3.metric("Killed", f"{filtered['n_killed'].sum():,}")
fatality_rate = filtered["n_killed"].sum() / filtered["n_people"].sum() * 100 if filtered["n_people"].sum() else 0
col4.metric("Fatality rate", f"{fatality_rate:.2f} %")

st.divider()

st.subheader("Accident map")
map_df = filtered.dropna(subset=["lat", "long"]).copy()
map_df["severity_label"] = map_df["accident_severity_class"].map(severity_labels)
if len(map_df) > 20000:
    map_df = map_df.sample(20000, random_state=0)  # cap for browser rendering, not the analysis
fig_map = px.scatter_map(
    map_df, lat="lat", lon="long", color="severity_label",
    color_discrete_map={"Fatal": "#d62728", "Hospitalised": "#ff7f0e", "Light injury": "#1f77b4"},
    hover_data={"dep": True, "date": True, "lat": False, "long": False},
    zoom=4.5, height=550, opacity=0.6,
)
fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Monthly severity trend")
    monthly = (
        filtered.groupby(["month", "accident_severity_class"]).size()
        .reset_index(name="n_accidents")
    )
    monthly["accident_severity_class"] = monthly["accident_severity_class"].map(severity_labels)
    fig_trend = px.bar(
        monthly, x="month", y="n_accidents", color="accident_severity_class",
        color_discrete_map={"Fatal": "#d62728", "Hospitalised": "#ff7f0e", "Light injury": "#1f77b4"},
        labels={"month": "Month", "n_accidents": "Accidents", "accident_severity_class": "Severity"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.subheader("Breakdown by cause (collision type)")
    causes = filtered["col"].map(col_labels).value_counts().reset_index()
    causes.columns = ["Collision type", "Accidents"]
    fig_causes = px.bar(causes.sort_values("Accidents"), x="Accidents", y="Collision type", orientation="h")
    st.plotly_chart(fig_causes, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Accidents by time of day")
    tod_order = ["night", "morning_rush", "midday", "evening_rush", "evening"]
    tod_labels = {"night": "Night", "morning_rush": "Morning rush", "midday": "Midday",
                  "evening_rush": "Evening rush", "evening": "Evening"}
    tod = filtered["time_of_day"].value_counts().reindex(tod_order).reset_index()
    tod.columns = ["time_of_day", "Accidents"]
    tod["Time of day"] = tod["time_of_day"].map(tod_labels)
    fig_tod = px.bar(tod, x="Time of day", y="Accidents")
    st.plotly_chart(fig_tod, use_container_width=True)

with chart_col4:
    st.subheader("Light conditions")
    lum = filtered["lum"].map(lum_labels).value_counts().reset_index()
    lum.columns = ["Light conditions", "Accidents"]
    fig_lum = px.pie(lum, names="Light conditions", values="Accidents")
    st.plotly_chart(fig_lum, use_container_width=True)

st.caption(
    "fact_accidents is at person x accident grain (125,187 rows, 54,402 accidents). "
    "The statistics above deduplicate on the accident before counting."
)
