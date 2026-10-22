import streamlit as st
from utils import *

######################################################################################
# Setup page
######################################################################################

st.set_page_config(layout="wide")

# Implement Session State
if "sku_master_df" not in st.session_state:
    st.session_state.sku_master_df = None

if "orders_df" not in st.session_state:
    st.session_state.orders_df = None

if "batches_df" not in st.session_state:
    st.session_state.batches_df = None

if "batches_df_fifo" not in st.session_state:
    st.session_state.batches_df_fifo = None

######################################################################################
# Session state functions
######################################################################################

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()

def reset_simulation():
    st.session_state.orders_df = None
    st.session_state.sku_master_df = None
    st.session_state.batches_df = None
    st.session_state.batches_df_fifo = None

    if "results" in st.session_state:
        del st.session_state["results"]

######################################################################################
# Header
######################################################################################

st.title("Picking Process - Optimization")
st.write("### Generate automatic batches and minimize picking distance")

######################################################################################

st.divider()

######################################################################################
# Step 1 - Create Orders
######################################################################################

st.write("### 1. Creating Orders")

with st.expander("Parameters"):

    st.caption('Feel free to adjust parameters to test different cases')

    par_1, par_2, par_3 = st.columns(3)

    with par_1:
        no_of_orders = st.number_input(
             label='No. of orders to create', 
             min_value=25, 
             max_value=150, 
             value=100, 
             step=10, 
             key='no_of_orders_key', 
             on_change=reset_simulation) 
    
    with par_2:
            no_of_kg = st.number_input(
                 label='Kg capacity per cart', 
                 min_value=25.0, 
                 max_value=150.0, 
                 value=75.0, 
                 step=5.0, 
                 key='no_of_kg_key', 
                 on_change=reset_simulation) 

    with par_3:
            no_of_cbm = st.number_input(
                 label='CBM capacity per cart',
                   min_value=25.0, 
                   max_value=150.0, 
                   value=100.0, 
                   step=10.0, 
                   key='no_of_cbm_key', 
                   on_change=reset_simulation) 
    
if st.button("Create orders", type='primary'):

    st.session_state.batches_df = None
    st.session_state.batches_df_fifo = None

    st.session_state.sku_master_df = create_sku_master(
        no_of_unique_zones=12,
        skus_per_zone=5,
        distribution_of_skus="balanced",
    )

    st.session_state.orders_df = create_orders(
        sku_master_df=st.session_state.sku_master_df,
        no_of_orders=no_of_orders,
        items_per_order_range=(1, 5),
        qty_range_per_item=(1, 5),
        volume_range_per_item=(0.02, 0.08),
        weight_range_per_item=(1.0, 3.0),
        random_state=42,
    )

if st.session_state.orders_df is not None:
        with st.expander("See created orders"):
                     st.caption("These are the orders that you have created synthetically.")
                     st.dataframe(st.session_state.orders_df, hide_index=True)
             


st.divider()

######################################################################################
# Step 2 - Create Batches
######################################################################################

st.write("### 2. Creating Optimized Batches")

if st.button(
    "Create optimized batches",
    disabled=st.session_state.orders_df is None,
    type='primary'
):

    merged_df = st.session_state.orders_df.merge(
        st.session_state.sku_master_df,
        on="SKU",
        how="left",
    )

    vector_matrix = (
        merged_df.pivot_table(
            index="Order_ID",
            columns="Zone",
            values="Qty",
            aggfunc="sum",
        )
        .fillna(0)
        .astype(int)
    )

    best_k, clustered_matrix, _ = find_optimal_spatial_clusters(
        vector_matrix,
        min_clusters=2,
        max_clusters=5,
    )

    pre_batches_df, oversized_df = capacity_bin_packing(
        orders_df=st.session_state.orders_df,
        clustered_matrix=clustered_matrix,
        max_cbm=no_of_cbm,
        max_kg=no_of_kg,
    )

    st.session_state.batches_df = optimize_residual_batches(
        batches_df=pre_batches_df,
        orders_df=st.session_state.orders_df,
        max_cbm=no_of_cbm,
        max_kg=no_of_kg,
        percentile_cutoff=20.0,
        utilization_cap=0.60,
    )

    pre_batches_df_fifo, oversized_df_fifo = fifo_bin_packing(
        orders_df=st.session_state.orders_df,
        max_cbm=no_of_cbm,
        max_kg=no_of_kg,
    )

    st.session_state.batches_df_fifo = optimize_residual_batches(
        batches_df=pre_batches_df_fifo,
        orders_df=st.session_state.orders_df,
        max_cbm=no_of_cbm,
        max_kg=no_of_kg,
        percentile_cutoff=20.0,
        utilization_cap=0.60,
    )

    # Add utilization factors
    st.session_state.batches_df["KG_Utilization"] = (
        st.session_state.batches_df["Total_KG"] / no_of_kg
        )*100

    st.session_state.batches_df["CBM_Utilization"] = (
        st.session_state.batches_df["Total_CBM"] / no_of_cbm
        )*100

    st.session_state.batches_df_fifo["KG_Utilization"] = (
        st.session_state.batches_df_fifo["Total_KG"] / no_of_kg
        )*100

    st.session_state.batches_df_fifo["CBM_Utilization"] = (
        st.session_state.batches_df_fifo["Total_CBM"] / no_of_cbm
        )*100
    
if st.session_state.batches_df is not None and st.session_state.batches_df_fifo is not None:
    with st.expander("Optimized Spatial Batches"):
         st.caption("These are batches that are grouped based on items that are physically close, thus minimizing the distances.")
         st.dataframe(st.session_state.batches_df, hide_index=True)

    with st.expander("Optimized FIFO Batches"):
        st.caption("These are batches that are grouped based on the arrival time of orders")
        st.dataframe(st.session_state.batches_df_fifo, hide_index=True)

    with st.expander("Utilization Insights"):
        st.caption("These are batches that are grouped based on xyz")

        col_4, col_5 = st.columns(2)

        with col_4:
            with st.container(border=True):
                fig_1 = plot_generic_comparison(
                df_primary=st.session_state.batches_df,
                df_baseline=st.session_state.batches_df_fifo,
                primary_col="KG_Utilization",
                baseline_col="KG_Utilization",
                primary_label="Opt. Clusters",
                baseline_label="FIFO Opt",
                metric_name="Utilization %",
                title="Utilization Overview (KG): Opt. Clusters vs. Opt. FIFO",
                )
                st.plotly_chart(fig_1)

        with col_5:
            with st.container(border=True):
                fig_2 = plot_generic_comparison(
                df_primary=st.session_state.batches_df,
                df_baseline=st.session_state.batches_df_fifo,
                primary_col="CBM_Utilization",
                baseline_col="CBM_Utilization",
                primary_label="Opt. Clusters",
                baseline_label="FIFO Opt",
                metric_name="Utilization %",
                title="Utilization Overview (CBM): Opt. Clusters vs. Opt. FIFO",
                )
                st.plotly_chart(fig_2)

st.divider()

######################################################################################
# Step 3 - Optimize Route
######################################################################################

st.write("### 3. Optimizing Picking Route")

if st.button(
    "Optimize picking route",
    disabled=st.session_state.batches_df is None,
    type='primary'
):

    distance_matrix, distance_df, coords = create_warehouse_distances(
        sku_master_df=st.session_state.sku_master_df,
        depot_name="Location_1",
    )

    routed_batches_df = route_capacitated_batches(
        batches_df=st.session_state.batches_df,
        orders_df=st.session_state.orders_df,
        distance_matrix=distance_matrix,
    )

    routed_batches_df_fifo = route_capacitated_batches(
        batches_df=st.session_state.batches_df_fifo,
        orders_df=st.session_state.orders_df,
        distance_matrix=distance_matrix,
    )

    d = {
        "optimized": routed_batches_df["Route_Distance_Meters"].sum(),
        "fifo": routed_batches_df_fifo["Route_Distance_Meters"].sum(),
    }

    winner = min(d, key=d.get)

    all_routed_batches_df = {
        "optimized": routed_batches_df,
        "fifo": routed_batches_df_fifo,
    }

######################################################################################
# Step 4 - Generate Metrics
######################################################################################

    benchmark_df = evaluate_tsp_performance(
        routed_batches_df=all_routed_batches_df[winner],
        orders_df=st.session_state.orders_df,
        distance_matrix=distance_matrix,
        depot_name="Location_1",
        num_simulations=100,
        seconds_per_meter=1.25,
    )

    total_opt_dist = benchmark_df["Optimal_TSP_Dist_Meters"].sum()
    total_base_dist = benchmark_df["Monte_Carlo_Baseline_Meters"].sum()
    total_dist_saved = benchmark_df["Distance_Saved_Meters"].sum()
    overall_pct_saved = (total_dist_saved / total_base_dist) * 100
    total_time_saved_min = benchmark_df["Time_Saved_Minutes"].sum()

    with st.expander("See Optimization Results"):
         col_6, col_7, col_8 = st.columns(3)

         with col_6:
            with st.container(border=True):
                 fig_3 = compare_route_distances(
                      routed_batches_df=all_routed_batches_df["optimized"], 
                      routed_batches_df_fifo=all_routed_batches_df["fifo"]
                )
                 st.plotly_chart(fig_3)
         
         with col_7:
              with st.container(border=True):
                st.write("##### --- Overall Fleet Evaluation ---")
                #st.write(f"Best Batch Type: {winner}")
                st.write(f"Best Batch Type: {'Optimized Cluster Batches' if winner == 'optimized' else 'Optimized FIFO Batches'}")
                st.write(f"Total Batches Processed: {len(benchmark_df)}")
                st.write(f"Un-optimized Walking Distance: {total_base_dist:,.2f} meters")
                st.write(f"Optimized TSP Distance: {total_opt_dist:,.2f} meters")
                st.write(f"Total Distance Saved: {total_dist_saved:,.2f} meters")
                st.success(f"Overall Fleet Efficiency Gain: {overall_pct_saved:.2f}%")
                st.success(
                    f"Total Picker Time Saved: {total_time_saved_min:.2f} minutes "
                    f"({total_time_saved_min/60:.2f} hours)"
                )

         with col_8:
            with st.container(border=True):
                 fig_4 = plot_generic_comparison(
                    df_primary=benchmark_df,
                    primary_col="Optimal_TSP_Dist_Meters",
                    baseline_col="Monte_Carlo_Baseline_Meters",
                    primary_label="Opt. Clusters + TSP",
                    baseline_label="Monte Carlo Random Path",
                    metric_name="Distance (Mts.)",
                    title="Optimized Distances (TSP Solver) vs. Non-Optimized Distances (Monte Carlo Baseline Distance)",
                )

                 st.plotly_chart(fig_4)
                 
    # Run post-processing
    batch_summary_df, pick_instructions_df = create_batch_pick_instructions(
        routed_batches_df=all_routed_batches_df[winner],
        orders_df=st.session_state.orders_df,
        sku_master_df=st.session_state.sku_master_df,
        depot_name="Location_1",
    )

    with st.expander("See Benchmark"):
        st.dataframe(batch_summary_df, hide_index=True)

    with st.expander("See Picking Instructions"):
            st.dataframe(pick_instructions_df, hide_index=True)
