# imports

import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import altair as alt
from libpysal.weights import DistanceBand
from shapely import Polygon
from concave_hull import concave_hull


# read in data

@st.cache_data(show_spinner='Loading data...')
def load_data():

    stations_data = gpd.read_file('data/stations_data_for_parameterized_problem_stations.geojson').set_index('station_id')
    
    tracts = gpd.read_file('data/tract_data.geojson')

    service_area = gpd.GeoDataFrame(
        geometry=[(
            stations_data.geometry
            .buffer(500)
            .unary_union
        )],
        crs=2263
    )

    tracts_in_service_area = (
        tracts
        .sjoin(service_area)
    )

    return stations_data, tracts, service_area, tracts_in_service_area

stations_data, tracts, service_area, tracts_in_service_area = load_data()

# TODO: display distributions of measures

# create slider selectors to take thresholds

freq_threshold = st.slider(
    label='frequency station has no bikes or no docks',
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    format='%d%%'
) / 100

freq_hist = (
    stations_data
    [['pct_am_or_evening_no_bikes_or_no_docks']]
    .assign(
        over_threshold = np.where(stations_data['pct_am_or_evening_no_bikes_or_no_docks'].ge(freq_threshold), 'selected','not selected')
    )
    .pipe(alt.Chart)
    .mark_bar()
    .encode(
        x=alt.X(
            'pct_am_or_evening_no_bikes_or_no_docks', 
            bin=alt.Bin(maxbins=20),
            axis=alt.Axis(format='%')
        ),
        y=alt.Y(
            'count(pct_am_or_evening_no_bikes_or_no_docks)',
            axis=None
            ),
        color=alt.Color(
            'over_threshold',
            legend=alt.Legend(orient='bottom-right', title='')
        )
    )
) 

freq_rule = (
    alt.Chart(
        pd.DataFrame({'selection':[freq_threshold]})
    )
    .mark_rule(color='red', strokeDash=[3, 2])
    .encode(
        x=alt.X('selection')
    )
)

freq_hist_display = (freq_hist + freq_rule).properties(
    height=250
)

st.altair_chart(freq_hist_display, use_container_width=True, theme=None)

st.write(f"freq threshold selects {stations_data['pct_am_or_evening_no_bikes_or_no_docks'].ge(freq_threshold).mean():.1%} of tations")

duration_threshold = st.slider(
    label='median duration of no bikes or no docks (minutes)',
    min_value=0.0,
    max_value=stations_data[['zero_bike_daytime_duration_median','zero_dock_daytime_duration_median']].max().max() * 60,
    value=60.0,
    step=5.0,
) / 60

duration_hist = (
    stations_data
    [['zero_bike_daytime_duration_median','zero_dock_daytime_duration_median']]
    .max(axis=1)
    .rename('duration')
    .to_frame()
    .assign(
        over_threshold = np.where(stations_data[['zero_bike_daytime_duration_median','zero_dock_daytime_duration_median']].max(axis=1).ge(duration_threshold), 'selected','not selected')
    )
    .pipe(alt.Chart)
    .mark_bar()
    .encode(
        x=alt.X(
            'duration', 
            bin=alt.Bin(maxbins=70),
            axis=None
        ),
        y=alt.Y(
            'count(duration)',
            axis=None
        ),
        color=alt.Color(
            'over_threshold',
            legend=alt.Legend(orient='bottom-right', title='')
        )
    )
) 

duration_rule = (
    alt.Chart(
        pd.DataFrame({'selection':[duration_threshold]})
    )
    .mark_rule(color='red', strokeDash=[3, 2])
    .encode(
        x=alt.X('selection')
    )
)

duration_hist_display = (duration_hist + duration_rule).properties(
    height=150
)

st.altair_chart(duration_hist_display, use_container_width=True, theme=None)

st.write(f"median duration threshold selects {((stations_data['zero_bike_daytime_duration_median'].ge(duration_threshold)) | (stations_data['zero_dock_daytime_duration_median'].ge(duration_threshold))).mean():.1%} of stations")

disabled_threshold = st.slider(
    label='pct of bikes at station disabled',
    min_value=0.0,
    max_value=stations_data['pct_of_docks_w_disabled_bikes_mean'].max() * 100,
    value=10.0,
    step=1.0
) / 100

disabled_hist = (
    stations_data
    [['pct_of_docks_w_disabled_bikes_mean']]
    .assign(
        over_threshold = np.where(stations_data['pct_of_docks_w_disabled_bikes_mean'].ge(disabled_threshold), 'selected','not selected')
    )
    .pipe(alt.Chart)
    .mark_bar()
    .encode(
        x=alt.X(
            'pct_of_docks_w_disabled_bikes_mean', 
            bin=alt.Bin(maxbins=30),
            axis=alt.Axis(format='%')
        ),
        y=alt.Y(
            'count(pct_of_docks_w_disabled_bikes_mean)',
            axis=None
        ),
        color=alt.Color(
            'over_threshold',
            legend=alt.Legend(orient='bottom-right', title='')
        )
    )
) 

disabled_rule = (
    alt.Chart(
        pd.DataFrame({'selection':[disabled_threshold]})
    )
    .mark_rule(color='red', strokeDash=[3, 2])
    .encode(
        x=alt.X('selection')
    )
)

disabled_hist_display = (disabled_hist + disabled_rule).properties(
    height=150
)

st.altair_chart(disabled_hist_display, use_container_width=True, theme=None)

st.write(f"bikes disabled threshold selects {stations_data['pct_of_docks_w_disabled_bikes_mean'].ge(disabled_threshold).mean():.1%} of stations")

# filter df to stations above thresholds

parameter_problem_stations = (
    stations_data
    [
        (stations_data['pct_am_or_evening_no_bikes_or_no_docks'].ge(freq_threshold)) |
        (stations_data['zero_bike_daytime_duration_median'].ge(duration_threshold)) |
        (stations_data['zero_dock_daytime_duration_median'].ge(duration_threshold)) |
        (stations_data['pct_of_docks_w_disabled_bikes_mean'].ge(disabled_threshold))
    ]
)

# annouce how much of data was filtered

st.write(f"combined, thresholds select {parameter_problem_stations.shape[0] / stations_data.shape[0]:.1%} of stations")


# filter to groups of >=5 stations

w_split_at_1320 = DistanceBand.from_dataframe(
    df=parameter_problem_stations,
    threshold=1320,
    binary=True
)

parameter_problem_stations = (
    parameter_problem_stations
    .assign(
        component = w_split_at_1320.component_labels
    )
)

parameter_problem_stations__component_5_or_more_nodes = (
    parameter_problem_stations
    [
        parameter_problem_stations
        ['component']
        .isin(
            parameter_problem_stations
            ['component']
            .value_counts()
            .ge(5)
            .where(lambda a:a).dropna()
            .index
        )
    ]
)

# draw concave hulls

problem_area_hulls = []

for component in parameter_problem_stations__component_5_or_more_nodes['component'].unique():

    component_geom = (
        parameter_problem_stations__component_5_or_more_nodes
        [
            parameter_problem_stations__component_5_or_more_nodes['component'] == component
        ]
        .geometry
    )

    component_xy = np.stack([
        component_geom.x.values,
        component_geom.y.values
    ]).T

    component_hull = concave_hull(
        component_xy,
        concavity=1.5
        )
    
    component_polygon = Polygon(component_hull)

    problem_area_hulls.append(Polygon(component_hull))

parameter_problem_areas = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(problem_area_hulls),
    crs=2263
)

parameter_problem_areas_buffer = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(problem_area_hulls).buffer(500),
    crs=2263
)

# sjoin to census tracts

tracts_in_parameter_problem_areas_buffer = (
    tracts
    .sjoin(
        parameter_problem_areas_buffer
    )
)

# display map

m = (
    tracts_in_service_area
    .explore(
        tiles='CartoDB Positron',
        color='gray',
        tooltip=None
    )
)

folium.GeoJson(
    tracts_in_parameter_problem_areas_buffer,
    style_function=lambda a: {'fillColor':'orange','color':'orange'}
).add_to(m)

folium.GeoJson(
    parameter_problem_areas_buffer,
    style_function=lambda a: {'fillColor':'firebrick','color':'firebrick'}
).add_to(m)


m = (
    stations_data
    .explore(
        m=m,
        tiles='CartoDB Positron',
        tooltip=None,
        color='white',
        marker_kwds=dict(
            radius=1.2,
            fill=True
        ),
        style_kwds=dict(
            #weight=1
            stroke=False,
            fillOpacity=0.9
            # color='white'
        )
    )
)

st_folium(m, returned_objects=[])

# display comparison demographics

comparison_demog = (
    tracts_in_service_area
    [[
        'White__2020', 
        'Black__2020', 
        'Asian__2020', 
        'Hispanic__2020',
        'below_150_pct_poverty_level__ACS_2021'
    ]]
    .sum()
    .div(
        tracts_in_service_area
        ['Total population__2020']
        .sum()
    )
    .rename('entire service area')
    .to_frame()
    .join(
        tracts_in_parameter_problem_areas_buffer
        [[
            'White__2020', 
            'Black__2020', 
            'Asian__2020', 
            'Hispanic__2020',
            'below_150_pct_poverty_level__ACS_2021'
        ]]
        .sum()
        .div(
            tracts_in_parameter_problem_areas_buffer
            ['Total population__2020']
            .sum()
        )
        .rename('poor service areas')
    )
    .reset_index()
    .melt(
        id_vars='index',
        var_name='area',
        value_name='pct of population'
    )
    .replace({
        
        'White__2020':'White (pct)', 
        'Black__2020':'Black (pct)', 
        'Asian__2020':'Asian (pct)', 
        'Hispanic__2020':'Hispanic/Latino (pct)',
        'below_150_pct_poverty_level__ACS_2021':'up to 150 pct of poverty level (pct)'
    })
    .rename(columns={'index':'race/demographic'})
)

# chart = (
#     alt.Chart(comparison_demog)
#     .mark_bar()
#     .encode(
#         y=alt.Y(
#             'area',
#             axis=None
#         ),
#         x='pct of population',
#         color=alt.Color(
#             'area',
#             scale=alt.Scale(
#                 domain=['entire service area','poor service areas'],
#                 range=['gray','firebrick']
#             )
#         ),
#         row= alt.Row(
#             'race/demographic',
#             title="",
#             header=alt.Header(
#                 labelAngle=0,
#                 labelAlign='right',
#                 labelAnchor='start',
#                 labelOrient='left'
#             )
#         )
#     )
# )

chart = (
    alt.Chart(comparison_demog)
    .mark_bar()
    .encode(
        x='race/demographic',
        y=alt.Y(
            'pct of population',
            axis=alt.Axis(format='%')
        ),
        color=alt.Color(
            'area',
            scale=alt.Scale(
                domain=['entire service area','poor service areas'],
                range=['gray','orange']
            ),
        ),
        xOffset='area'
    )
  
)


st.altair_chart(chart)

make_static_maps = st.button('make static maps')

if make_static_maps:

    with st.spinner():

        import contextily as cx
        import matplotlib.pyplot as plt

        fig,((black, hispanic), (asian, white), (poverty, income)) = plt.subplots(ncols=2,nrows=3, figsize=(12,33), dpi=200)

        (
            tracts_in_service_area
            .plot(
                ax=black,
                column='Black__2020__pct',
                cmap='Purples',
                scheme='NaturalBreaks',
                k=6,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Pct Black',
                    # bbox_to_anchor=(0,1),
                    fmt='{:.0%}'
                )
            )
        )

        (
            tracts_in_service_area
            .plot(
                ax=hispanic,
                column='Hispanic__2020__pct',
                cmap='RdPu',
                scheme='NaturalBreaks',
                k=6,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Pct Hispanic/Latino',
                    # bbox_to_anchor=(0,1),
                    fmt='{:.0%}'
                )
            )
        )

        (
            tracts_in_service_area
            .plot(
                ax=asian,
                column='Asian__2020__pct',
                cmap='Greens',
                scheme='NaturalBreaks',
                k=6,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Pct Asian',
                    # bbox_to_anchor=(0,1),
                    fmt='{:.0%}'
                )
            )
        )

        (
            tracts_in_service_area
            .plot(
                ax=white,
                column='White__2020__pct',
                cmap='Blues',
                scheme='NaturalBreaks',
                k=6,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Pct White',
                    # bbox_to_anchor=(0,1),
                    fmt='{:.0%}'
                )
            )
        )

        (
            tracts_in_service_area
            .plot(
                ax=poverty,
                column='below_150_pct_poverty_level__ACS_2021__pct',
                cmap='Reds',
                scheme='NaturalBreaks',
                k=6,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Pct <150 pct poverty',
                    # bbox_to_anchor=(0,1),
                    fmt='{:.0%}'
                )
            )
        )

        (
            tracts_in_service_area
            .plot(
                ax=income,
                column='median_income__ACS_2021',
                cmap='Spectral',
                scheme='Quantiles',
                k=5,
                alpha=0.8,
                legend=True,
                legend_kwds=dict(
                    title='Median income',
                    # bbox_to_anchor=(0,1),
                    fmt='${:,.0f}'
                )
            )
        )

        for ax in [black, hispanic, asian, white, poverty, income]:
            
            (
                stations_data
                .plot(
                    ax=ax,
                    markersize=0.5,
                    alpha=0.5,
                    color='lightgrey'
                )
            )

            (
                parameter_problem_areas_buffer
                .plot(
                    ax=ax,
                    edgecolor='orange',
                    facecolor='none',
                    legend=True,
                    linewidth=2.5,
                    linestyle=(0,(6,1))
                )
            )

            cx.add_basemap(
                ax, 
                crs=2263,
                source=cx.providers.CartoDB.PositronNoLabels
            )

            ax.axis('off')

        st.pyplot(fig, use_container_width=True)
