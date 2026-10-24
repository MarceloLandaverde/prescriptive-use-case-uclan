import pandas as pd
import string
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import pulp
import time
import plotly.express as px
import plotly.graph_objects as go

###############################################################################
# Master Data Genaration Functions
###############################################################################

def create_sku_master(
    no_of_unique_zones: int = 3,
    skus_per_zone: int = 2,
    distribution_of_skus: str = "balanced",
) -> pd.DataFrame:
    """Generates a structured SKU Master DataFrame based on the following format:

    [SKU_A1, SKU_A2 -> Zone_1], [SKU_B1, SKU_B2 -> Zone_2], etc.

    Parameters:
    - no_of_unique_zones (int): Total number of zones (1 to 26 mapped to
    letters A-Z).
    - skus_per_zone (int): Default number of SKUs per zone if balanced.
    - distribution_of_skus (str):
        * 'balanced': Equal number of SKUs per zone.
        * 'skewed': Front zones get significantly more SKUs than back zones.

    Returns:
    - pd.DataFrame with columns ['SKU', 'Zone']
    """
    skus = []
    zones = []

    alphabet = string.ascii_uppercase  # A, B, C...

    for j in range(no_of_unique_zones):
        zone_letter = alphabet[j % len(alphabet)]
        zone_name = f"Zone_{j+1}"

        # Determine SKU count for this zone based on distribution mode
        if distribution_of_skus == "balanced":
            count = skus_per_zone
        elif distribution_of_skus == "skewed":
            # Zone 1 gets the most, decreasing progressively
            count = max(1, int(skus_per_zone * (no_of_unique_zones - j)))
        else:
            raise ValueError("distribution_of_skus must be 'balanced' or 'skewed'")

        for k in range(1, count + 1):
            sku_name = f"SKU_{zone_letter}{k}"
            skus.append(sku_name)
            zones.append(zone_name)

    return pd.DataFrame({"SKU": skus, "Zone": zones})
#-------------------------------------------------------------------------------------

def create_orders(
    sku_master_df: pd.DataFrame,
    no_of_orders: int = 6,
    items_per_order_range: tuple = (1, 3),
    qty_range_per_item: tuple = (1, 2),
    volume_range_per_item: tuple = (0.01, 0.05),  # in CBM
    weight_range_per_item: tuple = (0.5, 2.5),  # in KG
    distribution_of_skus: str = "uniform",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates synthetic orders containing valid SKUs from sku_master_df along with quantity, volume (cbm), and weight (kg) measurements.

    Parameters:
    - sku_master_df (pd.DataFrame): The master dataframe generated in the function create_sku_master
    - no_of_orders (int): Total number of distinct customer orders to
    generate.
    - items_per_order_range (tuple): (min, max) number of distinct SKUs per
    order.
    - qty_range_per_item (tuple): (min, max) item quantity per SKU line.
    - volume_range_per_item (tuple): (min, max) volume per individual unit (in
    cbm).
    - weight_range_per_item (tuple): (min, max) weight per individual unit (in
    kg).
    - distribution_of_skus (str):
        * 'uniform': Equal likelihood of any SKU being picked.
        * 'popular_skus': Fast-movers (first 20% of SKUs) appear more
        frequently.
    - random_state (int): Seed for reproducibility.

    Returns:
    - pd.DataFrame with columns ['Order_ID', 'SKU', 'Qty', 'Unit_CBM',
    'Unit_KG', 'Total_CBM', 'Total_KG']
    """
    rng = np.random.default_rng(random_state)
    available_skus = sku_master_df["SKU"].unique()
    num_skus = len(available_skus)

    if num_skus == 0:
        raise ValueError("sku_master_df does not contain any valid SKUs.")

    # Assign physical properties (CBM and KG) to each SKU systematically
    sku_properties = {}
    for sku in available_skus:
        sku_properties[sku] = {
            "Unit_CBM": round(
                rng.uniform(volume_range_per_item[0], volume_range_per_item[1]),
                4,
            ),
            "Unit_KG": round(
                rng.uniform(weight_range_per_item[0], weight_range_per_item[1]),
                2,
            ),
        }

    # SKU selection probabilities
    if distribution_of_skus == "uniform":
        probabilities = None
    elif distribution_of_skus == "popular_skus":
        # First 20% of SKUs get 70% of total order demand (Pareto principle)
        weights = np.ones(num_skus)
        top_20_percent_count = max(1, int(num_skus * 0.2))
        weights[:top_20_percent_count] *= 5.0
        probabilities = weights / weights.sum()
    else:
        raise ValueError(
            "distribution_of_skus must be 'uniform' or 'popular_skus'"
        )

    order_rows = []

    for i in range(1, no_of_orders + 1):
        order_id = f"O{i}"

        # Determine number of distinct SKUs for this order
        num_items = rng.integers(
            items_per_order_range[0], items_per_order_range[1] + 1
        )
        num_items = min(num_items, num_skus)

        # Pick unique SKUs for this order
        selected_skus = rng.choice(
            available_skus, size=num_items, replace=False, p=probabilities
        )

        for sku in selected_skus:
            qty = int(
                rng.integers(qty_range_per_item[0], qty_range_per_item[1] + 1)
            )
            unit_cbm = sku_properties[sku]["Unit_CBM"]
            unit_kg = sku_properties[sku]["Unit_KG"]

            order_rows.append(
                {
                    "Order_ID": order_id,
                    "SKU": sku,
                    "Qty": qty,
                    "Unit_CBM": unit_cbm,
                    "Unit_KG": unit_kg,
                    "Total_CBM": round(qty * unit_cbm, 4),
                    "Total_KG": round(qty * unit_kg, 2),
                }
            )

    return pd.DataFrame(order_rows)
#--------------------------------------------------------------------------------------------------------------
def create_warehouse_distances(
    sku_master_df: pd.DataFrame,
    layout_type: str = "grid",  # 'grid' or 'asymmetric_grid'
    cross_aisle_dist: float = 5.0,  # Distance between neighboring zones (meters)
    aisle_length: float = 12.0,  # Total length of an aisle (meters)
    depot_name: str = "Location_1",  # Matches your PuLP depot naming convention
    depot_location: tuple = (0.0, 0.0),
    one_way_penalty: float = 0.0,  # Added distance for one-way traffic violation
) -> tuple[dict, pd.DataFrame, dict]:
    """Generates physical location mappings and a nested pairwise distance matrix formatted directly for PuLP MTZ lookup: distance_matrix[i][j].

    Returns:
    - distance_matrix (dict): Nested dictionary {i: {j: dist}} matching your exact PuLP input structure.
    - distance_df (pd.DataFrame): Distance matrix formatted as an N x N DataFrame.
    - location_coords (dict): Map of location nodes to (x, y) spatial coordinates.
    """
    # 1. Build List of Nodes (Depot first, followed by SKUs/Locations)
    unique_skus = sku_master_df["SKU"].unique().tolist()
    N = [depot_name] + unique_skus
    n_nodes = len(N)

    # 2. Assign Spatial Coordinates (x, y)
    location_coords = {depot_name: depot_location}

    for idx, row in sku_master_df.iterrows():
        sku = row["SKU"]
        zone_num = int(row["Zone"].split("_")[1])
        sku_num = int("".join(filter(str.isdigit, sku)))

        # X = Zone position across cross-aisle, Y = Depth down the aisle
        x = (zone_num - 1) * cross_aisle_dist
        y = sku_num * (aisle_length / 3.0)

        location_coords[sku] = (round(x, 2), round(y, 2))

    # 3. Construct Nested Distance Matrix matching distance_matrix[i][j]
    distance_matrix = {i: {} for i in N}
    matrix_array = np.zeros((n_nodes, n_nodes))

    for i_idx, i in enumerate(N):
        for j_idx, j in enumerate(N):
            if i == j:
                dist = 0
            else:
                x1, y1 = location_coords[i]
                x2, y2 = location_coords[j]

                # Standard Manhattan Distance
                manhattan_dist = abs(x1 - x2) + abs(y1 - y2)

                # Optional Asymmetric One-Way Directionality Penalty
                if (
                    layout_type == "asymmetric_grid"
                    and i != depot_name
                    and j != depot_name
                ):
                    zone_i = int(
                        sku_master_df.loc[
                            sku_master_df["SKU"] == i, "Zone"
                        ].values[0].split("_")[1]
                    )

                    # Even zones flow North (+Y), Odd zones flow South (-Y)
                    if zone_i % 2 == 0 and y2 < y1:
                        manhattan_dist += one_way_penalty
                    elif zone_i % 2 != 0 and y2 > y1:
                        manhattan_dist += one_way_penalty

                dist = int(round(manhattan_dist))

            distance_matrix[i][j] = dist
            matrix_array[i_idx, j_idx] = dist

    distance_df = pd.DataFrame(matrix_array, index=N, columns=N, dtype=int)

    return distance_matrix, distance_df, location_coords

###############################################################################
#Clustering Functions
###############################################################################

def find_optimal_spatial_clusters(
    vector_matrix: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = None,
    linkage: str = "ward",
) -> tuple[int, pd.DataFrame, dict]:
    """Evaluates multiple cluster counts using Silhouette Score and returns the best-fit AgglomerativeClustering model & cluster labels.

    Parameters:
    - vector_matrix (pd.DataFrame): Order-by-Zone quantity matrix.
    - min_clusters (int): Minimum number of clusters to evaluate (default: 2).
    - max_clusters (int): Maximum clusters to evaluate. If None, defaults to
    N_orders - 1.
    - linkage (str): Linkage criterion for AgglomerativeClustering ('ward',
    'complete', 'average').

    Returns:
    - optimal_k (int): Cluster count yielding the highest Silhouette Score.
    - vector_matrix_clustered (pd.DataFrame): Input matrix with 'Cluster' column
    appended.
    - scores (dict): Mapping of {k: silhouette_score}.
    """
    X = np.ascontiguousarray(vector_matrix.to_numpy(), dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples < 3:
        # Not enough orders to evaluate multiple clusters meaningfully
        vector_matrix_clustered = vector_matrix.copy()
        vector_matrix_clustered["Cluster"] = 0
        return 1, vector_matrix_clustered, {1: 1.0}

    # Set upper bound for max_clusters search
    if max_clusters is None or max_clusters >= n_samples:
        max_clusters = n_samples - 1

    min_clusters = max(2, min_clusters)

    best_k = min_clusters
    best_score = -1.0
    scores = {}

    # Evaluate each K in range
    for k in range(min_clusters, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)

        # Silhouette score requires at least 2 distinct clusters in output
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            scores[k] = round(score, 4)

            if score > best_score:
                best_score = score
                best_k = k

    # Fit optimal model
    optimal_model = AgglomerativeClustering(
        n_clusters=best_k, linkage=linkage
    )
    vector_matrix_clustered = vector_matrix.copy()
    vector_matrix_clustered["Cluster"] = optimal_model.fit_predict(X)

    return best_k, vector_matrix_clustered, scores

###############################################################################
#Capacity Functions#
###############################################################################

def capacity_bin_packing(
    orders_df: pd.DataFrame,
    clustered_matrix: pd.DataFrame,
    max_cbm: float = 0.35,  # Volume capacity limit
    max_kg: float = 20.0,  # Weight capacity limit
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits spatially clustered orders into distinct pick cart runs by enforcing

    physical capacity limits (Volume CBM, Weight KG). This is similar to a Knapsack Problem.

    Quarantines orders exceeding physical limits for specialized handling.

    Parameters:
    - orders_df (pd.DataFrame): Orders containing Qty, Total_CBM, and Total_KG.
    - clustered_matrix (pd.DataFrame): Vector matrix with 'Cluster' column.
    - max_cbm (float): Maximum total volume (cbm) per cart run.
    - max_kg (float): Maximum total weight (kg) per cart run.

    Returns:
    - batches_df (pd.DataFrame): Valid cart batches ready for routing.
    - oversized_df (pd.DataFrame): Quarantined orders exceeding cart capacity.
    """
    # 1. Aggregate total metrics per order
    order_totals = orders_df.groupby("Order_ID").agg(
        Order_Qty=("Qty", "sum"),
        Order_CBM=("Total_CBM", "sum"),
        Order_KG=("Total_KG", "sum"),
    )

    # Combine cluster assignment with order totals
    orders_with_metrics = clustered_matrix[["Cluster"]].join(order_totals)

    # 2. Separate Oversized Orders from Standard Orders (Physical Constraints Only)
    oversized_mask = (orders_with_metrics["Order_CBM"] > max_cbm) | (
        orders_with_metrics["Order_KG"] > max_kg
    )

    oversized_orders = orders_with_metrics[oversized_mask].copy()
    valid_orders = orders_with_metrics[~oversized_mask].copy()

    # Build summary table for quarantined oversized orders
    oversized_rows = []
    for order_id, row in oversized_orders.iterrows():
        reasons = []
        if row["Order_CBM"] > max_cbm:
            reasons.append(f"CBM ({row['Order_CBM']:.4f} > {max_cbm})")
        if row["Order_KG"] > max_kg:
            reasons.append(f"KG ({row['Order_KG']:.2f} > {max_kg})")

        oversized_rows.append(
            {
                "Order_ID": order_id,
                "Cluster": row["Cluster"],
                "Order_Qty": row["Order_Qty"],
                "Order_CBM": round(row["Order_CBM"], 4),
                "Order_KG": round(row["Order_KG"], 2),
                "Violation_Reason": ", ".join(reasons),
            }
        )

    oversized_df = pd.DataFrame(
        oversized_rows,
        columns=[
            "Order_ID",
            "Cluster",
            "Order_Qty",
            "Order_CBM",
            "Order_KG",
            "Violation_Reason",
        ],
    )

    # 3. Process Standard Valid Orders per Cluster
    final_batches = []
    batch_counter = 1

    for cluster_id, group in valid_orders.groupby("Cluster"):
        # Sort within cluster descending by volume (primary bottleneck)
        sorted_orders = group.sort_values(by="Order_CBM", ascending=False)

        current_batch_orders = []
        current_qty = 0.0
        current_cbm = 0.0
        current_kg = 0.0

        for order_id, row in sorted_orders.iterrows():
            o_qty = row["Order_Qty"]
            o_cbm = row["Order_CBM"]
            o_kg = row["Order_KG"]

            # Check if adding this order violates physical capacity constraints
            if (current_cbm + o_cbm > max_cbm) or (current_kg + o_kg > max_kg):

                # Close current batch and start a new one
                if current_batch_orders:
                    final_batches.append(
                        {
                            "Batch_ID": f"Batch_{batch_counter:02d}",
                            "Cluster": cluster_id,
                            "Orders": current_batch_orders,
                            "Total_Orders": len(current_batch_orders),
                            "Total_Qty": current_qty,
                            "Total_CBM": round(current_cbm, 4),
                            "Total_KG": round(current_kg, 2),
                        }
                    )
                    batch_counter += 1

                current_batch_orders = [order_id]
                current_qty = o_qty
                current_cbm = o_cbm
                current_kg = o_kg

            else:
                # Add order to current batch
                current_batch_orders.append(order_id)
                current_qty += o_qty
                current_cbm += o_cbm
                current_kg += o_kg

        # Append final batch in cluster
        if current_batch_orders:
            final_batches.append(
                {
                    "Batch_ID": f"Batch_{batch_counter:02d}",
                    "Cluster": cluster_id,
                    "Orders": current_batch_orders,
                    "Total_Orders": len(current_batch_orders),
                    "Total_Qty": current_qty,
                    "Total_CBM": round(current_cbm, 4),
                    "Total_KG": round(current_kg, 2),
                }
            )
            batch_counter += 1

    batches_df = pd.DataFrame(final_batches)
    return batches_df, oversized_df
#-----------------------------------------------------------------------------
def fifo_bin_packing(
    orders_df: pd.DataFrame, max_cbm: float = 100.0, max_kg: float = 75.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies capacity-constrained bin packing to a baseline control group 
    using volume-descending sorting heuristics (First-Fit Decreasing) under 
    a single dummy cluster. 

    Note: While serving as the baseline comparison group, orders are processed 
    through the standard capacity bin-packing engine to ensure a fair, 
    optimized-capacity control group rather than a naive chronological queue.
    """
    # 1. Create a deterministic FIFO order matrix sorted by Order_ID
    unique_orders = sorted(orders_df["Order_ID"].unique())
    fifo_matrix = pd.DataFrame(
        {"Cluster": "FIFO_Baseline"}, index=unique_orders
    )

    # 2. Pass directly to capacity engine to ensure 100% logic alignment
    return capacity_bin_packing(
        orders_df=orders_df,
        clustered_matrix=fifo_matrix,
        max_cbm=max_cbm,
        max_kg=max_kg,
    )

###############################################################################
#Optimizing Capacity Functions
###############################################################################

def optimize_residual_batches(
    batches_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    max_cbm: float = 100,
    max_kg: float = 75.0,
    percentile_cutoff: float = 20.0,
    utilization_cap: float = 0.60,
) -> pd.DataFrame:
    """Identifies under-utilized tail batches based on multi-dimensional bottleneck

    utilization and repacks their orders into consolidated carts.

    Parameters:
    - batches_df (pd.DataFrame): Output from capacity_bin_packing.
    - orders_df (pd.DataFrame): Master orders DataFrame with Qty, Total_CBM,
    Total_KG.
    - max_cbm (float): Max volume limit per cart.
    - max_kg (float): Max weight limit per cart.
    - percentile_cutoff (float): Percentile threshold to define low utilization
    (default 20.0).
    - utilization_cap (float): Hard upper limit (0.0 to 1.0) to qualify as a
    residual cart (default 0.60).

    Returns:
    - optimized_batches_df (pd.DataFrame): Cleaned batches ready for routing.
    """
    if batches_df.empty:
        return batches_df

    df = batches_df.copy()

    # 1. Calculate Bottleneck Utilization Ratio per Batch
    df["CBM_Util"] = df["Total_CBM"] / max_cbm
    df["KG_Util"] = df["Total_KG"] / max_kg
    df["Bottleneck_Util"] = df[["CBM_Util", "KG_Util"]].max(axis=1)

    # 2. Determine Statistical Cutoff Threshold
    cutoff_val = np.percentile(df["Bottleneck_Util"], percentile_cutoff)
    effective_threshold = min(cutoff_val, utilization_cap)

    # 3. Flag Residual vs. Mainstream Batches
    residual_mask = df["Bottleneck_Util"] <= effective_threshold

    mainstream_batches = df[~residual_mask].copy()
    residual_batches = df[residual_mask].copy()

    # If no batches qualify as residuals, return original structure clean
    if residual_batches.empty:
        df.drop(
            columns=["CBM_Util", "KG_Util", "Bottleneck_Util"], inplace=True
        )
        return df

    # 4. Extract and Unpack Residual Orders
    residual_order_ids = []
    for orders_list in residual_batches["Orders"]:
        if isinstance(orders_list, list):
            residual_order_ids.extend(orders_list)
        else:
            residual_order_ids.extend(
                [o.strip() for o in str(orders_list).split(",")]
            )

    # Get aggregated metrics for residual orders from orders_df
    res_order_metrics = (
        orders_df[orders_df["Order_ID"].isin(residual_order_ids)]
        .groupby("Order_ID")
        .agg(
            Order_Qty=("Qty", "sum"),
            Order_CBM=("Total_CBM", "sum"),
            Order_KG=("Total_KG", "sum"),
        )
        .reset_index()
    )

    # Sort descending by volume for efficient packing
    res_order_metrics = res_order_metrics.sort_values(
        by="Order_CBM", ascending=False
    )

    # 5. Repack Residual Pool into Consolidated Batches
    consolidated_batches = []
    curr_orders = []
    curr_qty, curr_cbm, curr_kg = 0.0, 0.0, 0.0

    for _, row in res_order_metrics.iterrows():
        o_id = row["Order_ID"]
        o_qty = row["Order_Qty"]
        o_cbm = row["Order_CBM"]
        o_kg = row["Order_KG"]

        if (curr_cbm + o_cbm > max_cbm) or (curr_kg + o_kg > max_kg):
            if curr_orders:
                consolidated_batches.append(
                    {
                        "Cluster": "Residual_Consolidated",
                        "Orders": curr_orders,
                        "Total_Orders": len(curr_orders),
                        "Total_Qty": curr_qty,
                        "Total_CBM": round(curr_cbm, 4),
                        "Total_KG": round(curr_kg, 2),
                    }
                )
            curr_orders = [o_id]
            curr_qty, curr_cbm, curr_kg = o_qty, o_cbm, o_kg
        else:
            curr_orders.append(o_id)
            curr_qty += o_qty
            curr_cbm += o_cbm
            curr_kg += o_kg

    if curr_orders:
        consolidated_batches.append(
            {
                "Cluster": "Residual_Consolidated",
                "Orders": curr_orders,
                "Total_Orders": len(curr_orders),
                "Total_Qty": curr_qty,
                "Total_CBM": round(curr_cbm, 4),
                "Total_KG": round(curr_kg, 2),
            }
        )

    # 6. Re-assemble Final Cleaned Batches Table
    mainstream_clean = mainstream_batches[
        [
            "Cluster",
            "Orders",
            "Total_Orders",
            "Total_Qty",
            "Total_CBM",
            "Total_KG",
        ]
    ].copy()
    consolidated_df = pd.DataFrame(consolidated_batches)

    final_df = pd.concat(
        [mainstream_clean, consolidated_df], ignore_index=True
    )

    # Re-assign sequential Batch_IDs
    final_df["Batch_ID"] = [
        f"Batch_{i+1:02d}" for i in range(len(final_df))
    ]

    # Reorder columns to match standard schema
    cols = [
        "Batch_ID",
        "Cluster",
        "Orders",
        "Total_Orders",
        "Total_Qty",
        "Total_CBM",
        "Total_KG",
    ]
    return final_df[cols]

###############################################################################
#Optimization Functions
###############################################################################

def solve_2opt_heuristic(
    nodes: list, distance_matrix: dict
) -> tuple[list, float]:
    """Fast Nearest-Neighbor + 2-Opt heuristic for large batches (N > 12) otherwise the computation runtime would be too high.

    Executes in under 5 milliseconds.
    """
    unvisited = set(nodes)
    unvisited.remove("Location_1")
    curr = "Location_1"
    route = ["Location_1"]

    # 1. Nearest Neighbor Initial Tour
    while unvisited:
        nxt = min(unvisited, key=lambda loc: distance_matrix[curr][loc])
        route.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    route.append("Location_1")

    # 2. 2-Opt Local Search Improvement
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue
                # Calculate current edge costs vs reversed edge costs
                cost_curr = (
                    distance_matrix[route[i - 1]][route[i]]
                    + distance_matrix[route[j]][route[j + 1]]
                )
                cost_new = (
                    distance_matrix[route[i - 1]][route[j]]
                    + distance_matrix[route[i]][route[j + 1]]
                )

                if cost_new < cost_curr:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True

    # Calculate total tour length
    total_dist = sum(
        distance_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1)
    )
    return route, float(total_dist)

#-------------------------------------------------------------------------------------------
def solve_atsp_safe(
    nodes: list, distance_matrix: dict, max_pulp_nodes: int = 12, time_limit_sec: int = 5
) -> tuple[list, float]:
    """Solves TSP using PuLP for small N (exact solution), and automatically falls back to 2-Opt Heuristic for large N to prevent long runtimes.
    """
    n = len(nodes)

    if n <= 1:
        return ["Location_1", "Location_1"], 0.0
    if n == 2:
        other = [loc for loc in nodes if loc != "Location_1"][0]
        dist = distance_matrix["Location_1"][other] + distance_matrix[other]["Location_1"]
        return ["Location_1", other, "Location_1"], float(dist)

    # Use fast 2-Opt heuristic if node count is too large for exact MILP
    if n > max_pulp_nodes:
        return solve_2opt_heuristic(nodes, distance_matrix)

    # PuLP MTZ Formulation for small N
    model = pulp.LpProblem("Batch_TSP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", [(i, j) for i in nodes for j in nodes if i != j], cat=pulp.LpBinary
    )
    non_depot = [i for i in nodes if i != "Location_1"]
    u = pulp.LpVariable.dicts(
        "u", non_depot, lowBound=2, upBound=n, cat=pulp.LpInteger
    )

    # Objective
    model += pulp.lpSum(
        x[i, j] * distance_matrix[i][j] for i in nodes for j in nodes if i != j
    )

    # Constraints
    for i in nodes:
        model += pulp.lpSum(x[i, j] for j in nodes if j != i) == 1
    for j in nodes:
        model += pulp.lpSum(x[i, j] for i in nodes if i != j) == 1

    for i in non_depot:
        for j in non_depot:
            if i != j:
                model += u[j] - u[i] >= 1 - n * (1 - x[i, j])

    # Enforce solver timeout (timeLimit in seconds)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    model.solve(solver)

    # Extract Route
    succ = {}
    for i in nodes:
        for j in nodes:
            if i != j and pulp.value(x[i, j]) is not None:
                if pulp.value(x[i, j]) > 0.5:
                    succ[i] = j

    curr = "Location_1"
    route = ["Location_1"]
    while True:
        curr = succ.get(curr, "Location_1")
        route.append(curr)
        if curr == "Location_1":
            break

    total_dist = pulp.value(model.objective)
    return route, float(total_dist)
###############################################################################
# Wrapping Opt. Functions into Routing Functions
###############################################################################
def route_capacitated_batches(
    batches_df: pd.DataFrame, orders_df: pd.DataFrame, distance_matrix: dict
) -> pd.DataFrame:
    """Processes all batches from capacity_bin_packing and computes pick routes safely.
    """
    routed_rows = []
    start_time = time.time()

    for idx, batch in batches_df.iterrows():
        batch_id = batch["Batch_ID"]
        order_list = batch["Orders"]

        # Extract unique SKUs required by orders in this batch
        batch_skus = orders_df[orders_df["Order_ID"].isin(order_list)][
            "SKU"
        ].unique()
        nodes_to_visit = ["Location_1"] + list(batch_skus)

        # Solve safely (PuLP for N <= 12, 2-Opt for N > 12)
        route_seq, total_dist = solve_atsp_safe(
            nodes_to_visit, distance_matrix, max_pulp_nodes=12, time_limit_sec=3
        )

        routed_rows.append(
            {
                "Batch_ID": batch_id,
                "Cluster": batch["Cluster"],
                "Orders": order_list,
                "Total_Qty": batch["Total_Qty"],
                "Total_CBM": batch["Total_CBM"],
                "Total_KG": batch["Total_KG"],
                "Unique_Locations": len(batch_skus),
                "Pick_Route": " -> ".join(route_seq),
                "Route_Distance_Meters": total_dist,
            }
        )

    print(
        f"--- All {len(batches_df)} Batches Routed in {time.time() - start_time:.2f} seconds ---"
    )
    return pd.DataFrame(routed_rows)
#--------------------------------------------------------------------------------------------------
def create_batch_pick_instructions(
    routed_batches_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    sku_master_df: pd.DataFrame,
    depot_name: str = "Location_1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-processes routed batches into two structured DataFrames:

    1. batch_summary_df: High-level KPIs per batch (stops, distance, totals).
    2. pick_instructions_df: Granular stop-by-stop pick list with total batch quantity displayed at the final Depot unload stop.
    """
    sku_to_zone = dict(zip(sku_master_df["SKU"], sku_master_df["Zone"]))
    instruction_rows = []
    summary_rows = []

    for _, batch in routed_batches_df.iterrows():
        batch_id = batch["Batch_ID"]
        order_list = batch["Orders"]

        # 1. Filter order items belonging to this batch
        batch_items = orders_df[orders_df["Order_ID"].isin(order_list)].copy()
        batch_items["Zone"] = batch_items["SKU"].map(sku_to_zone)

        # Aggregate total quantity required per SKU in this batch
        sku_qty_map = (
            batch_items.groupby("SKU")["Qty"]
            .sum()
            .to_dict()
        )

        # 2. Extract SKU sequence from Pick_Route string
        raw_route = [node.strip() for node in batch["Pick_Route"].split("->")]

        stops = []
        stop_seq = 1

        # 3. Start Depot Stop (Cart is empty: Qty = 0)
        stops.append(
            {
                "Batch_ID": batch_id,
                "Stop_Seq": stop_seq,
                "Location_Type": "Depot",
                "Zone_Location": depot_name,
                "SKUs_to_Pick": "-",
                "Pick_Details": "Start Route / Collect Empty Cart",
                "Total_Qty_at_Stop": 0,
            }
        )

        visited_skus = [node for node in raw_route if node != depot_name]

        # Group consecutive or same-location SKUs into physical stops
        location_grouping = {}
        for sku in visited_skus:
            zone = sku_to_zone.get(sku, "Unknown_Zone")
            if zone not in location_grouping:
                location_grouping[zone] = []
            if sku not in location_grouping[zone]:
                location_grouping[zone].append(sku)

        # Track total route quantity picked
        total_batch_picked_qty = 0

        # 4. Create Pick Zone Stops
        for zone, skus in location_grouping.items():
            stop_seq += 1
            pick_details = ", ".join(
                [f"{sku} (x{sku_qty_map.get(sku, 0)})" for sku in skus]
            )
            total_stop_qty = sum(sku_qty_map.get(sku, 0) for sku in skus)
            total_batch_picked_qty += total_stop_qty

            stops.append(
                {
                    "Batch_ID": batch_id,
                    "Stop_Seq": stop_seq,
                    "Location_Type": "Pick Zone",
                    "Zone_Location": zone,
                    "SKUs_to_Pick": ", ".join(skus),
                    "Pick_Details": pick_details,
                    "Total_Qty_at_Stop": total_stop_qty,
                }
            )

        # 5. End Depot Stop (Unload total accumulated batch items)
        stop_seq += 1
        stops.append(
            {
                "Batch_ID": batch_id,
                "Stop_Seq": stop_seq,
                "Location_Type": "Depot",
                "Zone_Location": depot_name,
                "SKUs_to_Pick": "-",
                "Pick_Details": f"End Route / Unload {total_batch_picked_qty} Total Items at Drop-off",
                "Total_Qty_at_Stop": total_batch_picked_qty,
            }
        )

        instruction_rows.extend(stops)

        # Build summary row per batch
        summary_rows.append(
            {
                "Batch_ID": batch_id,
                "Cluster": batch["Cluster"],
                "Orders_Count": len(order_list),
                "Orders": ", ".join(map(str, order_list)),
                "Total_Qty": batch["Total_Qty"],
                "Total_CBM": batch["Total_CBM"],
                "Total_KG": batch["Total_KG"],
                "Physical_Stops": len(location_grouping),
                "Unique_SKUs": len(visited_skus),
                "Route_Distance_Meters": batch["Route_Distance_Meters"],
            }
        )

    pick_instructions_df = pd.DataFrame(instruction_rows)
    batch_summary_df = pd.DataFrame(summary_rows)

    return batch_summary_df, pick_instructions_df
###########################################################################
# Evaluation Functions
###########################################################################
def evaluate_tsp_performance(
    routed_batches_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    distance_matrix: dict,
    depot_name: str = "Location_1",
    num_simulations: int = 100,
    seconds_per_meter: float = 1.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluates TSP route performance against a Monte Carlo un-optimized baseline.

    Parameters:
    - routed_batches_df (pd.DataFrame): Batches processed by route_capacitated_batches.
    - orders_df (pd.DataFrame): Orders dataframe to extract batch SKUs.
    - distance_matrix (dict): Nested dictionary distance_matrix[i][j].
    - depot_name (str): Starting/ending location identifier.
    - num_simulations (int): Number of Monte Carlo random permutations per batch.
    - seconds_per_meter (float): Walking time conversion factor (default: 1.25 s/m).
    - random_state (int): Seed for reproducible Monte Carlo shuffling.

    Returns:
    - benchmark_df (pd.DataFrame): Performance breakdown per batch and total fleet summary.
    """
    rng = np.random.default_rng(random_state)
    benchmark_rows = []

    for _, batch in routed_batches_df.iterrows():
        batch_id = batch["Batch_ID"]
        order_list = batch["Orders"]
        optimal_dist = batch["Route_Distance_Meters"]

        # Extract unique SKUs required by this batch
        batch_skus = orders_df[orders_df["Order_ID"].isin(order_list)][
            "SKU"
        ].unique().tolist()
        n_locations = len(batch_skus)

        if n_locations <= 1:
            # Single location: baseline equals optimal route
            avg_baseline_dist = optimal_dist
        else:
            sim_distances = []
            for _ in range(num_simulations):
                # Randomly shuffle pick locations (keeping Depot fixed at start and end)
                shuffled_skus = batch_skus.copy()
                rng.shuffle(shuffled_skus)
                perm_route = [depot_name] + shuffled_skus + [depot_name]

                # Compute permuted route distance
                dist = sum(
                    distance_matrix[perm_route[k]][perm_route[k + 1]]
                    for k in range(len(perm_route) - 1)
                )
                sim_distances.append(dist)

            avg_baseline_dist = float(np.mean(sim_distances))

        # Calculate metrics
        dist_saved = max(0.0, avg_baseline_dist - optimal_dist)
        pct_savings = (
            (dist_saved / avg_baseline_dist * 100) if avg_baseline_dist > 0 else 0.0
        )
        time_saved_sec = dist_saved * seconds_per_meter

        benchmark_rows.append(
            {
                "Batch_ID": batch_id,
                "Unique_SKUs": n_locations,
                "Optimal_TSP_Dist_Meters": round(optimal_dist, 2),
                "Monte_Carlo_Baseline_Meters": round(avg_baseline_dist, 2),
                "Distance_Saved_Meters": round(dist_saved, 2),
                "Distance_Savings_Pct": round(pct_savings, 2),
                "Time_Saved_Seconds": round(time_saved_sec, 2),
                "Time_Saved_Minutes": round(time_saved_sec / 60.0, 2),
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows)

    return benchmark_df

###############################################################################
#Visualization Functions
###############################################################################

def plot_generic_comparison(
    df_primary: pd.DataFrame,
    df_baseline: pd.DataFrame = None,
    primary_col: str = "Route_Distance_Meters",
    baseline_col: str = "Monte_Carlo_Baseline_Meters",
    primary_label: str = "Primary Strategy",
    baseline_label: str = "Baseline Strategy",
    batch_id_col: str = "Batch_ID",
    metric_name: str = "Value",
    title: str = "",
) -> go.Figure:
    """Generates a generic Plotly boxplot comparing any numerical metric across two strategies.

    Accepts either two separate DataFrames or a single DataFrame. Uses native
    Plotly defaults.
    """
    # 1. Normalize Inputs
    primary_df = df_primary[[batch_id_col, primary_col]].copy()
    primary_df.rename(columns={primary_col: "Metric_Value"}, inplace=True)

    if df_baseline is not None:
        baseline_df = df_baseline[[batch_id_col, baseline_col]].copy()
        baseline_df.rename(columns={baseline_col: "Metric_Value"}, inplace=True)
    else:
        baseline_df = df_primary[[batch_id_col, baseline_col]].copy()
        baseline_df.rename(columns={baseline_col: "Metric_Value"}, inplace=True)

    # 2. Build Traces using Plotly Defaults
    fig = go.Figure()

    # Primary Strategy Trace
    fig.add_trace(
        go.Box(
            y=primary_df["Metric_Value"],
            x=[primary_label] * len(primary_df),
            name=primary_label,
            boxpoints="all",
            hovertext=primary_df[batch_id_col].astype(str),
        )
    )

    # Baseline Strategy Trace
    fig.add_trace(
        go.Box(
            y=baseline_df["Metric_Value"],
            x=[baseline_label] * len(baseline_df),
            name=baseline_label,
            boxpoints="all",
            hovertext=baseline_df[batch_id_col].astype(str),
        )
    )

    # 3. Format Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        yaxis=dict(title=metric_name),
        xaxis=dict(title="Strategy"),
        template="plotly_white",
        boxmode="group",
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50),
    )

    return fig



def compare_route_distances(routed_batches_df, routed_batches_df_fifo):
    """
    Compares the total route distance between the optimized pipeline 
    and the FIFO baseline using a Plotly bar chart.
    """
    # Calculate total distances
    opt_distance = routed_batches_df['Route_Distance_Meters'].sum()
    fifo_distance = routed_batches_df_fifo['Route_Distance_Meters'].sum()
    
    # Build a comparison DataFrame
    df_plot = pd.DataFrame({
        'Method': ['Opt. Spatial Batches + TSP', 'Opt. FIFO Batches'],
        'Total Distance (m)': [opt_distance, fifo_distance]
    })
    
    # Create Plotly bar chart
    fig = px.bar(
        df_plot,
        x='Method',
        y='Total Distance (m)',
        color='Method',
        title='Batch Routing Efficiency: Opt. Spatial Batches vs. Opt. FIFO Batches',
        text='Total Distance (m)',
        color_discrete_sequence=['#1f77b4', '#aec7e8']
    )
    
    # Format layout for presentation quality
    fig.update_traces(texttemplate='%{text:.1f} m', textposition='outside')
    fig.update_layout(
        xaxis_title='Batch Routing Strategy',
        yaxis_title='Total Distance (Meters)',
        template='plotly_white',
        showlegend=False,
        uniformtext_minsize=8, 
        uniformtext_mode='hide'
    )
    
    return fig








