import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import io
from collections import defaultdict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

################### BUILD ORDER_DEPTHS FROM DATAFRAME #######################

def build_order_depths(df: pd.DataFrame) -> dict:
    order_depths = defaultdict(dict)

    for _, row in df.iterrows():
        timestamp = str(row["timestamp"])
        product = row["product"]

        # Build buy_orders dictionary
        buy_orders = {}
        for i in range(1, 4):
            price = row.get(f"bid_price_{i}")
            volume = row.get(f"bid_volume_{i}")
            if pd.notna(price) and pd.notna(volume):
                buy_orders[int(price)] = int(volume)

        # Build sell_orders dictionary (volumes should be negative)
        sell_orders = {}
        for i in range(1, 4):
            price = row.get(f"ask_price_{i}")
            volume = row.get(f"ask_volume_{i}")
            if pd.notna(price) and pd.notna(volume):
                sell_orders[int(price)] = -int(volume)

        order_depths[timestamp][product] = OrderDepth(buy_orders, sell_orders)

    return dict(order_depths)

################### FILE READING  #######################
def create_agreggated_files(comp_round, base_path = rf"C:\Users\gonzaal\Desktop\Personal\Personal python projects\Prosperity 3"):

    # List to store the datafrakmes for all days
    all_market_data = []
    all_trade_history = []

    # Iterate iver the days you want to read
    for day in range(-2, 1):  
        prices_file = os.path.join(base_path, rf"round{comp_round}\data\prices_round_1_day_{day}.csv")
        trades_file = os.path.join(base_path, rf"round{comp_round}\data\trades_round_1_day_{day}.csv")

        try:
            market_data = pd.read_csv(prices_file, sep=";", header=0)
            all_market_data.append(market_data)
        except FileNotFoundError:
            print(f"File {prices_file} was not found\n")

        try:
            trade_history = pd.read_csv(trades_file, sep=";", header=0)
            all_trade_history.append(trade_history)
        except FileNotFoundError:
            print(f"File {trades_file} was not found\n{80*'-'}")

    # Concatena all dataframes 
    market_data_ALL = pd.concat(all_market_data, ignore_index=True)
    trade_history_ALL = pd.concat(all_trade_history, ignore_index=True)

    # Convert the concatenated dataframes to csv
    output_prices_file = os.path.join(base_path, rf"round{comp_round}\data\prices_round_{comp_round}_ALL.csv")
    output_trades_file = os.path.join(base_path, rf"round{comp_round}\data\trades_round_{comp_round}_ALL.csv")

    market_data_ALL.to_csv(output_prices_file, sep=";", index=False)
    trade_history_ALL.to_csv(output_trades_file, sep=";", index=False)

    print(f"Concatenated market data stored in: {output_prices_file}\n")
    print(f"Concatenated trade history stored in: {output_trades_file}")



def parse_backtester_output(file_path):
    """
    Parses the backtester output file into a market data DataFrame and a trade history list.
    
    The output file is expected to contain three sections:
      - Sandbox logs (ignored)
      - Activities log: a CSV section starting with the header line after "Activities log:"
      - Trade History: a JSON array starting after "Trade History:"
    
    Parameters:
      file_path (str): Path to the backtester output file.
    
    Returns:
      market_data (pd.DataFrame): The activities log parsed into a DataFrame.
      trades (list): The trade history parsed as a list of dictionaries.
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    # Split sections based on markers
    # We use the markers "Activities log:" and "Trade History:" as delimiters.
    # First, split off sandbox logs.
    parts = content.split("Activities log:")
    if len(parts) < 2:
        raise ValueError("Activities log section not found.")
    # The second part has CSV + other sections; now split the CSV and trade history.
    activities_and_rest = parts[1]
    activities_part, *rest = activities_and_rest.split("Trade History:")
    if not rest:
        raise ValueError("Trade History section not found.")
    trade_history_part = rest[0]
    
    # The Activities log portion is in CSV format.
    # Sometimes there might be leading/trailing whitespace or extra newlines.
    csv_text = activities_part.strip()
    # Read CSV using semicolon as delimiter.
    market_data = pd.read_csv(io.StringIO(csv_text), delimiter=";")
    
    # The Trade History portion is a JSON array.
    trade_history_text = trade_history_part.strip()
    # In case there are extra characters before/after the JSON (e.g. newlines), try to extract the JSON starting with '[':
    json_start = trade_history_text.find('[')
    if json_start == -1:
        raise ValueError("JSON start '[' not found in Trade History section.")
    json_text = trade_history_text[json_start:]
    trades = json.loads(json_text)
    
    return market_data, trades

################### DATA PLOTTING #######################

def plot_backtest_results(market_data: pd.DataFrame, trades: list):
    """
    Graph metrics for each product using market data and trade history.
    
    This function creates three subplots per product:
      1) Profit & Loss (PnL) over time.
      2) Cumulative position computed from the trade history.
      3) Trade execution prices, marking buys and sells.
    
    Parameters:
      market_data: DataFrame containing at least the following columns:
                   - 'timestamp': time of the market snapshot.
                   - 'product': product identifier.
                   - 'profit_and_loss': PnL at that snapshot.
      trades: List of trade dictionaries, each with keys:
              'timestamp', 'buyer', 'seller', 'symbol' (product), 'price', 'quantity'.
    """
    
    # Sort the market data by timestamp.
    market_data = market_data.sort_values('timestamp')
    
    # Extract the unique products (assumed to be in the 'product' column)
    products = market_data['product'].unique()
    n_products = len(products)
    
    # Create a figure with one row per product and three columns for each subplot.
    fig, axs = plt.subplots(n_products, 3, figsize=(18, 5 * n_products), squeeze=False)
    
    # Convert the trade list into a DataFrame and sort by timestamp.
    trades_df = pd.DataFrame(trades)
    trades_df.sort_values('timestamp', inplace=True)
    
    for i, product in enumerate(products):
        # Filter market_data and trades for the current product.
        prod_market = market_data[market_data['product'] == product]
        prod_trades = trades_df[trades_df['symbol'] == product].copy()
        
        # --- PnL Plot ---
        ax = axs[i][0]
        ax.plot(prod_market['timestamp'], prod_market['profit_and_loss'], marker='o', linestyle='-')
        ax.set_title(f"{product} - Profit & Loss")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PnL")
        ax.grid(True)
        
        # --- Cumulative Position Plot ---
        # We compute the position change based on trades.
        # We assume that when your algorithm is the buyer (with "SUBMISSION" on seller),
        # the position increases, and when it is the seller (with "SUBMISSION" on buyer), it decreases.
        def position_change(row):
            if row['seller'] == "SUBMISSION":
                return row['quantity']
            elif row['buyer'] == "SUBMISSION":
                return -row['quantity']
            else:
                return 0
        
        if not prod_trades.empty:
            prod_trades['position_change'] = prod_trades.apply(position_change, axis=1)
            prod_trades['cumulative_position'] = prod_trades['position_change'].cumsum()
        else:
            # If no trades exist for this product, create a dummy timeline with zero positions.
            prod_trades = pd.DataFrame({
                'timestamp': prod_market['timestamp'],
                'cumulative_position': 0
            })

        ax = axs[i][1]
        ax.plot(prod_trades['timestamp'], prod_trades['cumulative_position'], marker='o', linestyle='-', color='orange')
        ax.set_title(f"{product} - Cumulative Position")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Position")
        ax.grid(True)
        
        # --- Order / Trade Prices Plot ---
        # Separate the trade markers into buys and sells.
        buys = prod_trades[prod_trades['position_change'] > 0]
        sells = prod_trades[prod_trades['position_change'] < 0]

        ax = axs[i][2]
        if not buys.empty:
            ax.scatter(buys['timestamp'], buys['price'], marker='^', color='green', label="Buy")
        if not sells.empty:
            ax.scatter(sells['timestamp'], sells['price'], marker='v', color='red', label="Sell")
        ax.set_title(f"{product} - Order Execution Prices")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()



def calculate_market_spreads(market_data_path, write_csv = False, csv_path = None):

    df = pd.read_csv(market_data_path,sep=";",header=0)

    # Create a unique time index by combining day and timestamp
    df['time_index'] = df['day'].astype(str) + '-' + df['timestamp'].astype(str)

    for i in range(1,4):
        df[f'spread_{i}'] = np.where(df[f'ask_price_{i}'].notna() & df[f'bid_price_{i}'].notna(),df[f'ask_price_{i}'] - df[f'bid_price_{i}'], np.nan)

    # Calculate best_bid and best_ask
    df['ask_price_best'] = df[['ask_price_1', 'ask_price_2', 'ask_price_3']].min(axis=1)
    df['bid_price_best'] = df[['bid_price_1', 'bid_price_2', 'bid_price_3']].max(axis=1)

    # Tightes spread
    df['spread_best'] = np.where(df['ask_price_best'].notna() & df['bid_price_best'].notna(),
                            df['ask_price_best'] - df['bid_price_best'], np.nan)

    # Initialize columns for best bid/ask volumes
    df['ask_volume_best'] = 0
    df['bid_volume_best'] = 0

    # Determine the best bid/ask volume
    for index, row in df.iterrows():
        # Find the best ask volume
        if row['ask_price_best'] == row['ask_price_1']:
            df.at[index, 'ask_volume_best'] = row['ask_volume_1']
        elif row['ask_price_best'] == row['ask_price_2']:
            df.at[index, 'ask_volume_best'] = row['ask_volume_2']
        elif row['ask_price_best'] == row['ask_price_3']:
            df.at[index, 'ask_volume_best'] = row['ask_volume_3']

        # Find the best bid volume
        if row['bid_price_best'] == row['bid_price_1']:
            df.at[index, 'bid_volume_best'] = row['bid_volume_1']
        elif row['bid_price_best'] == row['bid_price_2']:
            df.at[index, 'bid_volume_best'] = row['bid_volume_2']
        elif row['bid_price_best'] == row['bid_price_3']:
            df.at[index, 'bid_volume_best'] = row['bid_volume_3']
    
    if (write_csv == True) and (csv_path is not None):
        df.to_csv(csv_path, index=False)

    return df


def plot_spreads_grid(df, spreads, colors={"KELP": "green", "RAINFOREST_RESIN": "red", "SQUID_INK": "purple"}):
    products = df['product'].unique()
    n_products = len(products)
    n_spreads = len(spreads)

    # Dynamic width: wider figure if only 1 spread
    fig_width = 16 if n_spreads == 1 else 12 * n_spreads
    fig_height = 10 * n_products  # Increased from 4 to 5
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = fig.add_gridspec(n_products, n_spreads, wspace=0.2, hspace=0.2)

    for i, product in enumerate(products):
        for j, spread in enumerate(spreads):
            product_data = df[df['product'] == product].copy()
            
            spread_str = str(spread)
            spread_col = f'spread_{spread_str}'
            bid_col = f'bid_volume_{spread_str}'
            ask_col = f'ask_volume_{spread_str}'

            if spread_col not in product_data.columns:
                print(f"Warning: {spread_col} not found for product {product}. Skipping.")
                continue

            product_data[f'{spread_col}_rolling'] = product_data[spread_col].rolling(window=300).mean()
            product_data[f'bid_volume_{spread_str}_rolling'] = product_data[bid_col].rolling(window=300).mean()
            product_data[f'ask_volume_{spread_str}_rolling'] = product_data[ask_col].rolling(window=300).mean()
            product_data['net_volume'] = product_data[ask_col] - product_data[bid_col]
            product_data['net_volume_rolling'] = product_data['net_volume'].rolling(window=300).mean()

            inner_gs = outer_gs[i, j].subgridspec(2, 1, height_ratios=[7, 2], hspace=0.15)
            ax_main = fig.add_subplot(inner_gs[0])
            ax_vol = fig.add_subplot(inner_gs[1], sharex=ax_main)

            color = colors.get(product, 'blue')
            ax_main.plot(product_data[f'{spread_col}_rolling'], label=f'{product} - Spread {spread_str}', color=color)
            ax_main.set_ylabel('Rolling Spread')
            ax_main.set_title(f'{product} - Spread {spread_str}')
            ax_main.legend(loc='upper left')
            plt.setp(ax_main.get_xticklabels(), visible=False)

            ax_vol.bar(product_data.index, product_data['net_volume_rolling'], color='black', alpha=0.5, label='Rolling Volume')
            ax_vol.set_ylabel('Volume')
            ax_vol.set_xlabel('Time')
            ax_vol.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
