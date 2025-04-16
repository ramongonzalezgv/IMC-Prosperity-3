
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, Tuple
import string
import jsonpickle
import numpy as np
from statistics import NormalDist
import math
import json




def BS_CALL(S, K, T, r, sig):
    N = NormalDist().cdf

    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)

    price = (S * N(d1) - K * np.exp(-r * T) * N(d2))

    return (price)



def delta(S, K, T, r, sig) -> float:
    """Delta of the option."""

    N = NormalDist().cdf
    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    
    delta = N(d1)

    return(delta)


def gamma(S, K, T, r, sig) -> float:
    """Gamma of the option."""

    N = NormalDist().cdf
    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))

    gamma = N(d1) / (S * sig * np.sqrt(T))

    return gamma


def vega(S, K, T, r, sig) -> float:
    """Vega of the option."""

    N = NormalDist().cdf
    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
  
    vega = S * N(d1) * np.sqrt(T) / 100

    return vega


def theta(S, K, T, r, sig) -> float:
    """Theta of the option."""

    N = NormalDist().cdf
    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    
    first_term = (-S * N(d1) * sig ) / (2 * np.sqrt(T))
    second_term = r * K * np.exp(-r * T) * N(d2)
    
    theta = (first_term - second_term) / 365

    return theta


def rho(S, K, T, r, sig) -> float:
    """Rho of the option."""

    N = NormalDist().cdf
    d1 = (np.log(S/K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)

    rho = (K * T * np.exp(-r * T) * N(d2)) / 100

def implied_volatility(
        Option_price, S, K, T, r, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BS_CALL(
                S, K, T,r, volatility
            )
            diff = estimated_price - Option_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility





class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()



class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    SYNTHETIC_1 = "SYNTHETIC_1"
    SPREAD_1 = "SPREAD_1"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": 25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
     Product.SPREAD: {
        "default_spread_mean": 22.5,
        "default_spread_std": 57,
        "spread_std_window": 25,
        "zscore_threshold": 7,
        "target_position": 0,
        "spread_history": [],
        "prev_zscore": 0       
    },
    Product.SPREAD_1: {
        "default_spread_mean": 57.6,
        "default_spread_std": 88.4,
        "spread_std_window": 25,
        "zscore_threshold": 7,
        "target_position": 58,
        "spread_history": [],
        "prev_zscore": 0       
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.180032,
        "std_volatility": 0.094180,
        "z_threshold": 1,
        "strike": 9500,
        "time to maturity": 5 / 252,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.190640,
        "std_volatility": 0.043150,
        "z_threshold": 1,
        "strike": 9750,
        "time to maturity": 5 / 252,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.174819,
        "std_volatility": 0.007611,
        "z_threshold": 1,
        "strike": 10000,
        "time to maturity": 5 / 252,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.159284,
        "std_volatility": 0.004646,
        "z_threshold": 1,
        "strike": 10250,
        "time to maturity": 5 / 252,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.155616,
        "std_volatility": 0.005521,
        "z_threshold": 1,
        "strike": 10500,
        "time to maturity": 5 / 252,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
}


BASKET_WEIGHTS = {
Product.CROISSANTS: 4,
Product.JAMS: 2,
}

BASKET_WEIGHTS_1 = {
Product.CROISSANTS: 6,
Product.JAMS: 3,
Product.DJEMBES: 1
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50, 
            Product.SQUID_INK: 50, 
            Product.CROISSANTS: 250,
            Product.JAMS:350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 600,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None
    
    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None
    
    def squid_ink_fair_value_rolling(
        self, 
        order_depth: OrderDepth,
        traderData: TradingState,
    ): 
        curr_mid_price = self.get_mid_price(order_depth)
        mid_price_history = traderData[Product.SQUID_INK]["mid price diff history"]

        mid_price_history.append(curr_mid_price)

        if (len(mid_price_history) < self.params[Product.SQUID_INK]["mid price diff window"]):

            return None
        
        elif (len(mid_price_history)> self.params[Product.SQUID_INK]["mid price diff window"]):

            mid_price_history.pop(0)

        mid_price_rolling_mean = np.mean(mid_price_history)
        mid_price_rolling_std = np.std(mid_price_history)

        return mid_price_rolling_mean, mid_price_rolling_std
    
    def squid_ink_take_orders(
        self,
        order_depth: OrderDepth,
        traderData: TradingState,
        squid_ink_position: int,
    ):
        
        squid_limit = self.LIMIT[Product.SQUID_INK]
        
        squid_ink_orders = []

        best_bid_price = max(order_depth.buy_orders.keys())
        best_ask_price = min(order_depth.sell_orders.keys())

        best_bid_vol = order_depth.buy_orders[best_bid_price]
        best_ask_vol = order_depth.sell_orders[best_ask_price]
        
        squid_ink_mid_price = self.get_mid_price(order_depth)
        squid_ink_fair_value = self.squid_ink_fair_value_rolling(order_depth,traderData)

        if squid_ink_fair_value:

            dif = squid_ink_fair_value - squid_ink_mid_price

            if dif > 50:
                if squid_ink_position < squid_limit:
                    quantity = min(squid_limit - squid_ink_position,best_ask_vol)
                    if quantity > 0:
                        squid_ink_orders.append(Order(Product.SQUID_INK,best_ask_price,quantity))
            if dif < -10:
                if squid_ink_position > -squid_limit:
                    quantity = min(squid_limit + squid_ink_position,best_bid_vol)
                    if quantity > 0:
                        squid_ink_orders.append(Order(Product.SQUID_INK,best_bid_price,-quantity))

        return squid_ink_orders

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def get_swmid_1(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
        )
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )

            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume)
            
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price
    def get_synthetic_basket_order_depth_1(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS_1[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS_1[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS_1[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
            + DJEMBES_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
            + DJEMBES_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume, DJEMBES_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price


    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )

            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
           
            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)

        return component_orders
    def convert_synthetic_basket_orders_1(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth_1(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS_1[Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS_1[Product.JAMS],
            )
            DJEMBES_order = Order(
                Product.DJEMBES, DJEMBES_price, quantity * BASKET_WEIGHTS_1[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders    
    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders
    def execute_spread_orders_1(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth_1(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC_1, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders_1(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC_1, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders_1(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
    
    
    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None
        
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        # spread_data["spread_history"].append(55)
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def spread_orders_1(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position_1: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth_1(order_depths)
        basket_swmid = self.get_swmid_1(basket_order_depth)
        synthetic_swmid = self.get_swmid_1(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD_1]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD_1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD_1]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD_1]["zscore_threshold"]:
            if basket_position_1 != -self.params[Product.SPREAD_1]["target_position"]:
                return self.execute_spread_orders_1(
                    -self.params[Product.SPREAD_1]["target_position"],
                    basket_position_1,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD_1]["zscore_threshold"]:
            if basket_position_1 != self.params[Product.SPREAD_1]["target_position"]:
                return self.execute_spread_orders_1(
                    self.params[Product.SPREAD_1]["target_position"],
                    basket_position_1,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def get_best_bid(self, order_depth):

        best_bid_price = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid_price])
        
        return (best_bid_price, best_bid_vol)
    
    def get_best_ask(self, order_depth):
    
        best_ask_price = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask_price])

        return(best_ask_price,best_ask_vol)
    
    def get_mid_price(self, order_depth):

        best_bid_price = max(order_depth.buy_orders.keys())
        best_ask_price = min(order_depth.sell_orders.keys())

        return (best_bid_price + best_ask_price)/2
    
    def get_mid_price2(self, order_depth):

        best_bid_price = max(order_depth.buy_orders.keys())
        best_ask_price = min(order_depth.sell_orders.keys())

        return (best_bid_price + best_ask_price)/2
        

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def VolcanicRockCoupon_strat(self, 
        order_depths: Dict[str, OrderDepth],
        prod,
        position: int
    ):
        
        option_orders = []
        
        if prod not in order_depths.keys() or len(order_depths[prod].buy_orders) == 0 or len(order_depths[prod].sell_orders) == 0:
            return option_orders
        
        VolcanicRock_best_bid_price, _ = self.get_best_bid(order_depths[Product.VOLCANIC_ROCK])
        VolcanicRock_best_ask_price, _ = self.get_best_ask(order_depths[Product.VOLCANIC_ROCK])
        
        VolcanicRock_mid_price = (VolcanicRock_best_bid_price + VolcanicRock_best_ask_price)/2

        option_best_bid_price, option_best_bid_volume = self.get_best_bid(order_depths[prod])
        option_best_ask_price, option_best_ask_volume = self.get_best_ask(order_depths[prod])
        

        K = self.params[prod]['strike']
        T = self.params[prod]['time to maturity']
        sig = self.params[prod]['mean_volatility']
        r = 0

        BS_price = BS_CALL(VolcanicRock_mid_price, K, T, r, sig)

        
        if BS_price > option_best_ask_price:
            if position < self.LIMIT[prod]:
                quantity = min(option_best_ask_volume, self.LIMIT[prod] - position)  
                option_orders.append(Order(prod,option_best_ask_price,quantity))

        elif BS_price < option_best_bid_price:
                if position > -self.LIMIT[prod]:
                    quantity = min(option_best_bid_volume, self.LIMIT[prod] + position)
                    option_orders.append(Order(prod,option_best_bid_price,-quantity))

        return option_orders
    
    def VolcanicRockCoupon_strat_Zscore(self, 
        order_depths: Dict[str, OrderDepth],
        prod,
        position: int
    ):
        
        option_orders = []
        
        if prod not in order_depths.keys() or len(order_depths[prod].buy_orders) == 0 or len(order_depths[prod].sell_orders) == 0:
            return option_orders
        
        VolcanicRock_best_bid_price, _ = self.get_best_bid(order_depths[Product.VOLCANIC_ROCK])
        VolcanicRock_best_ask_price, _ = self.get_best_ask(order_depths[Product.VOLCANIC_ROCK])
        
        VolcanicRock_mid_price = (VolcanicRock_best_bid_price + VolcanicRock_best_ask_price)/2

        option_best_bid_price, option_best_bid_volume = self.get_best_bid(order_depths[prod])
        option_best_ask_price, option_best_ask_volume = self.get_best_ask(order_depths[prod])
        

        K = self.params[prod]['strike']
        T = self.params[prod]['time to maturity']
        sig = self.params[prod]['mean_volatility']
        std_vol = self.params[prod]['std_volatility']
        r = 0
        z_threshold = std_vol = self.params[prod]['z_threshold']
        

        BS_price = BS_CALL(VolcanicRock_mid_price, K, T, r, sig)

        
        if std_vol > 0 and (BS_price - option_best_ask_price) / std_vol > z_threshold:
            if position < self.LIMIT[prod]:
                quantity = min(option_best_ask_volume, self.LIMIT[prod] - position)  
                option_orders.append(Order(prod,option_best_ask_price,quantity))

        elif std_vol > 0 and (BS_price - option_best_bid_price) / std_vol < -z_threshold:
                if position > -self.LIMIT[prod]:
                    quantity = min(option_best_bid_volume, self.LIMIT[prod] + position)
                    option_orders.append(Order(prod,option_best_bid_price,-quantity))

        return option_orders
    
    def hedge_option_orders(
        self,
        underlying_prod,
        order_depths: Dict[str, OrderDepth],
        option_orders: List[Order],
        prod_position: int,
        option_position: int,
        delta: float
    ) -> List[Order]:
        
        buy_order_quantity = 0
        sell_order_quantity = 0
        
        orders: List[Order] = []
        
        if option_orders == None or len(option_orders) == 0:
            option_position_after_trade = option_position
        else:
            option_position_after_trade = option_position + sum(order.quantity for order in option_orders)
        
        target_underlying_position = -delta * option_position_after_trade
        
        if target_underlying_position == prod_position:
            return orders
        
        target_underlying_quantity = target_underlying_position - prod_position

        
        if target_underlying_quantity > 0:
            # Buy underlying
            best_ask = min(order_depths[underlying_prod].sell_orders.keys())
            quantity = min(
                abs(target_underlying_quantity),
                self.LIMIT[underlying_prod] - prod_position,
            )
            if quantity > 0:
                orders.append(Order(underlying_prod, best_ask, round(quantity)))
        
        elif target_underlying_quantity < 0:
            # Sell underlying
            best_bid = max(order_depths[underlying_prod].buy_orders.keys())
            quantity = min(
                abs(target_underlying_quantity),
                self.LIMIT[underlying_prod] + prod_position,
            )
            if quantity > 0:
                orders.append(Order(underlying_prod, best_bid, -round(quantity)))
        
        return orders
    
    def hedge_option_orders_optimized(
        self,
        underlying_prod,
        order_depths: Dict[str, OrderDepth],
        option_orders: List[Order],
        prod_position: int,
        option_position: int,
        delta: float
    ) -> List[Order]:
        
        buy_order_quantity = 0
        sell_order_quantity = 0
        
        hedge_orders: List[Order] = []
        
        if option_orders == None or len(option_orders) == 0:
            option_position_after_trade = option_position
        else:
            option_position_after_trade = option_position + sum(order.quantity for order in option_orders)
        
        target_underlying_position = -delta * option_position_after_trade
        
        if target_underlying_position == prod_position:
            option_orders = []
            return option_orders, hedge_orders
        
        target_underlying_quantity = target_underlying_position - prod_position

        
        if target_underlying_quantity > 0:
            # Buy underlying
            best_ask = min(order_depths[underlying_prod].sell_orders.keys())
            quantity = min(
                abs(target_underlying_quantity),
                self.LIMIT[underlying_prod] - prod_position,
            )
            if quantity > 0:
                hedge_orders.append(Order(underlying_prod, best_ask, round(quantity)))
        
        elif target_underlying_quantity < 0:
            # Sell underlying
            best_bid = max(order_depths[underlying_prod].buy_orders.keys())
            quantity = min(
                abs(target_underlying_quantity),
                self.LIMIT[underlying_prod] + prod_position,
            )
            if quantity > 0:
                hedge_orders.append(Order(underlying_prod, best_bid, -round(quantity)))
        
        return option_orders, hedge_orders
    
    def vertical_spread_pair_strat(
        self,
        order_depths: Dict[str, OrderDepth],
        position: Dict[str, int],
        lower_prod: str,
        higher_prod: str,
        threshold: float = 1.0
    ) -> Tuple[List[Order], List[Order]]:
        """
        Create a 1:1 call vertical spread between lower_prod and higher_prod.
        Returns two lists: (lower_leg_orders, higher_leg_orders).
        """
        lower_orders: List[Order] = []
        higher_orders: List[Order] = []

        # 1) Ensure both legs have valid order books
        for p in (lower_prod, higher_prod):
            if (p not in order_depths or
                not order_depths[p].buy_orders or
                not order_depths[p].sell_orders):
                return lower_orders, higher_orders

        # 2) Underlying mid‐price
        ub, _ = self.get_best_bid(order_depths[Product.VOLCANIC_ROCK])
        ua, _ = self.get_best_ask(order_depths[Product.VOLCANIC_ROCK])
        S = (ub + ua) / 2

        # 3) Market prices & vols for both options
        l_ask, l_ask_vol = self.get_best_ask(order_depths[lower_prod])
        l_bid, l_bid_vol = self.get_best_bid(order_depths[lower_prod])
        h_ask, h_ask_vol = self.get_best_ask(order_depths[higher_prod])
        h_bid, h_bid_vol = self.get_best_bid(order_depths[higher_prod])

        # 4) Theoretical Black‐Scholes prices
        r = 0
        T = self.params[lower_prod]['time to maturity']   # same for both
        sigma = self.params[lower_prod]['mean_volatility']
        K1 = self.params[lower_prod]['strike']
        K2 = self.params[higher_prod]['strike']

        theo_l = BS_CALL(S, K1, T, r, sigma)
        theo_h = BS_CALL(S, K2, T, r, sigma)
        theo_spread = theo_l - theo_h

        # 5) Market‐observed spreads
        market_bull_spread = l_ask - h_bid   # cost to buy lower & sell higher
        market_bear_spread = l_bid - h_ask   # credit for selling lower & buying higher

        # 6) Bull call spread?
        if theo_spread - market_bull_spread > threshold:
            qty = min(
                l_ask_vol,
                h_bid_vol,
                self.LIMIT[lower_prod] - position.get(lower_prod, 0),
                self.LIMIT[higher_prod] + position.get(higher_prod, 0),
            )
            if qty > 0:
                lower_orders.append( Order(lower_prod,  l_ask,  qty) )   # buy lower strike
                higher_orders.append(Order(higher_prod, h_bid, -qty))   # sell higher strike

        # 7) Bear call spread?
        elif market_bear_spread - theo_spread > threshold:
            qty = min(
                l_bid_vol,
                h_ask_vol,
                self.LIMIT[lower_prod] + position.get(lower_prod, 0),
                self.LIMIT[higher_prod] - position.get(higher_prod, 0),
            )
            if qty > 0:
                lower_orders.append( Order(lower_prod,  l_bid, -qty) )   # sell lower strike
                higher_orders.append(Order(higher_prod, h_ask,  qty) )  # buy higher strike

        return higher_orders, lower_orders
    
    def hedge_with_delta_gamma(
        self,
        underlying_prod,
        main_prod,  # primary option product for which you've taken a position
        main_prod_option_orders, # List of option orders to be hedged
        hedging_prod,  # secondary option product used for gamma hedging
        order_depths: dict,
        underlying_position,  # current underlying position
    ):
        
        Q1 = 0 # current primary option position quantity (signed: positive for long, negative for short)
        for order in main_prod_option_orders:
            Q1 += order.quantity

        # Get mid-price for underlying
        best_bid_underlying, _ = self.get_best_bid(order_depths[underlying_prod])
        best_ask_underlying, _ = self.get_best_ask(order_depths[underlying_prod])
        S = (best_bid_underlying + best_ask_underlying) / 2

        r = 0  # risk-free rate assumption

        # Extract parameters for main and hedging option
        K1 = self.params[main_prod]['strike']
        T1 = self.params[main_prod]['time to maturity']
        sigma1 = self.params[main_prod]['mean_volatility']
        
        K2 = self.params[hedging_prod]['strike']
        T2 = self.params[hedging_prod]['time to maturity']
        sigma2 = self.params[hedging_prod]['mean_volatility']

        # Compute greeks for the primary option
        delta1 = delta(S, K1, T1, r, sigma1)
        gamma1 = gamma(S, K1, T1, r, sigma1)

        # Compute greeks for the hedging option
        delta2 = delta(S, K2, T2, r, sigma2)
        gamma2 = gamma(S, K2, T2, r, sigma2)

        # Determine the quantity Q2 required for gamma neutrality
        # Q1 * gamma1 + Q2 * gamma2 = 0  => Q2 = - (Q1 * gamma1) / gamma2
        if gamma2 == 0:
            Q2 = 0
        else:
            Q2 = - (Q1 * gamma1) / gamma2

        # Now compute the net delta exposure after the two option positions:
        net_option_delta = Q1 * delta1 + Q2 * delta2

        # The underlying position needed for delta neutrality:
        # net_option_delta + Q3 = 0  => Q3 = -net_option_delta
        Q3 = - net_option_delta

        # Create orders for hedging option and underlying
        hedge_option_orders = []
        underlying_hedge_orders = []

        # Order for hedging option
        if Q2 > 0:
            # Need to buy hedging options
            best_ask_hedge, _ = self.get_best_ask(order_depths[hedging_prod])
            hedge_option_orders.append(Order(hedging_prod, best_ask_hedge, round(Q2)))
        elif Q2 < 0:
            best_bid_hedge, _ = self.get_best_bid(order_depths[hedging_prod])
            hedge_option_orders.append(Order(hedging_prod, best_bid_hedge, -round(Q2)))

        # Order for underlying asset
        if Q3 > 0:
            best_ask_underlying, _ = self.get_best_ask(order_depths[underlying_prod])
            # Ensure that you respect position limits if applicable:
            quantity = min(round(Q3), self.LIMIT[underlying_prod] - underlying_position)
            if quantity > 0:
                underlying_hedge_orders.append(Order(underlying_prod, best_ask_underlying, quantity))
        elif Q3 < 0:
            best_bid_underlying, _ = self.get_best_bid(order_depths[underlying_prod])
            quantity = min(round(abs(Q3)), self.LIMIT[underlying_prod] + underlying_position)
            if quantity > 0:
                underlying_hedge_orders.append(Order(underlying_prod, best_bid_underlying, -quantity))
        
        return hedge_option_orders, underlying_hedge_orders
        
        


    def run(self, state: TradingState):

        
    
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)


        traderObject = json.loads(state.traderData) if state.traderData != "" else {}
        result = {}
        conversions = 0


        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )

            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )

            result[Product.SQUID_INK] = (SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders )

            # SQUID_INK_take_orders = self.squid_ink_take_orders(state.order_depths[Product.SQUID_INK],traderObject,SQUID_INK_position)

            # result[Product.SQUID_INK] = SQUID_INK_take_orders

        # if Product.SPREAD1 not in traderObject:
        #     traderObject[Product.SPREAD1] = {
        #         "spread_history": []
        #     }


        # BASKT = self.baskets_products
        # basket_position = state.position[Product.PICNIC_BASKET1] if Product.PICNIC_BASKET1 in state.position else 0
        # spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderObject[Product.SPREAD1]["spread_history"])
        # if spread_orders != None:
        #     result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
        #     result[Product.JAMS] = spread_orders[Product.JAMS]
        #     result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
        #     result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        #########################################################################################################################################################################

        # option_product = Product.VOLCANIC_ROCK_VOUCHER_9500

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys():

        #     option_mid_price = (min(state.order_depths[option_product].buy_orders.keys())+ max(state.order_depths[option_product].sell_orders.keys())) / 2
        #     underlying_mid_price = (min(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys())+ max(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys())) / 2
            

        #     VOLCANIC_ROCK_VOUCHER_10000_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
            

        #     VOLCANIC_ROCK_position = (
        #         state.position[Product.VOLCANIC_ROCK]
        #         if Product.VOLCANIC_ROCK in state.position
        #         else 0
        #     )

        #     K = self.params[option_product]['strike']
        #     T = self.params[option_product]['time to maturity']
        #     r = 0

        #     volatility = implied_volatility(
        #         option_mid_price,
        #         underlying_mid_price,
        #         self.params[option_product]["strike"],
        #         T,
        #         r
        #     )

        #     delta1 = delta(underlying_mid_price,K,T,r,volatility)

        #     option_orders, hedge_orders = self.hedge_option_orders_optimized(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

        #     result[Product.VOLCANIC_ROCK] = hedge_orders
        #     result[option_product] = option_orders

        # #########################################################################################################################################################################

        # option_product = Product.VOLCANIC_ROCK_VOUCHER_9750

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys():

        #     option_mid_price = (min(state.order_depths[option_product].buy_orders.keys())+ max(state.order_depths[option_product].sell_orders.keys())) / 2
        #     underlying_mid_price = (min(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys())+ max(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys())) / 2
            

        #     VOLCANIC_ROCK_VOUCHER_10000_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)


        #     VOLCANIC_ROCK_position = (
        #         state.position[Product.VOLCANIC_ROCK]
        #         if Product.VOLCANIC_ROCK in state.position
        #         else 0
        #     )

        #     K = self.params[option_product]['strike']
        #     T = self.params[option_product]['time to maturity']
        #     r = 0

        #     volatility = implied_volatility(
        #         option_mid_price,
        #         underlying_mid_price,
        #         self.params[option_product]["strike"],
        #         T,
        #         r
        #     )

        #     delta1 = delta(underlying_mid_price,K,T,r,volatility)

        #     option_orders, hedge_orders = self.hedge_option_orders_optimized(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

        #     result[option_product] = option_orders
        #     result[Product.VOLCANIC_ROCK] = hedge_orders
            

        # #########################################################################################################################################################################

        # option_product = Product.VOLCANIC_ROCK_VOUCHER_10000

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys():

        #     option_mid_price = (min(state.order_depths[option_product].buy_orders.keys())+ max(state.order_depths[option_product].sell_orders.keys())) / 2
        #     underlying_mid_price = (min(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys())+ max(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys())) / 2
            

        #     VOLCANIC_ROCK_VOUCHER_10000_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
            

        #     VOLCANIC_ROCK_position = (
        #         state.position[Product.VOLCANIC_ROCK]
        #         if Product.VOLCANIC_ROCK in state.position
        #         else 0
        #     )

        #     K = self.params[option_product]['strike']
        #     T = self.params[option_product]['time to maturity']
        #     r = 0

        #     volatility = implied_volatility(
        #         option_mid_price,
        #         underlying_mid_price,
        #         self.params[option_product]["strike"],
        #         T,
        #         r
        #     )

        #     delta1 = delta(underlying_mid_price,K,T,r,volatility)

        #     option_orders, hedge_orders = self.hedge_option_orders_optimized(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

        #     result[option_product] = option_orders
        #     result[Product.VOLCANIC_ROCK] = hedge_orders

        # #########################################################################################################################################################################

        # option_product = Product.VOLCANIC_ROCK_VOUCHER_10250

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys():

        #     option_mid_price = (min(state.order_depths[option_product].buy_orders.keys())+ max(state.order_depths[option_product].sell_orders.keys())) / 2
        #     underlying_mid_price = (min(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys())+ max(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys())) / 2
            

        #     VOLCANIC_ROCK_VOUCHER_10000_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
            

        #     VOLCANIC_ROCK_position = (
        #         state.position[Product.VOLCANIC_ROCK]
        #         if Product.VOLCANIC_ROCK in state.position
        #         else 0
        #     )

        #     K = self.params[option_product]['strike']
        #     T = self.params[option_product]['time to maturity']
        #     r = 0

        #     volatility = implied_volatility(
        #         option_mid_price,
        #         underlying_mid_price,
        #         self.params[option_product]["strike"],
        #         T,
        #         r
        #     )

        #     delta1 = delta(underlying_mid_price,K,T,r,volatility)

        #     option_orders, hedge_orders = self.hedge_option_orders_optimized(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

        #     result[Product.VOLCANIC_ROCK] = hedge_orders
        #     result[option_product] = option_orders

        # #########################################################################################################################################################################

        # option_product = Product.VOLCANIC_ROCK_VOUCHER_10500

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys() and state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys():

        #     option_mid_price = (min(state.order_depths[option_product].buy_orders.keys())+ max(state.order_depths[option_product].sell_orders.keys())) / 2
        #     underlying_mid_price = (min(state.order_depths[Product.VOLCANIC_ROCK].buy_orders.keys())+ max(state.order_depths[Product.VOLCANIC_ROCK].sell_orders.keys())) / 2
            

        #     VOLCANIC_ROCK_VOUCHER_10000_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)

        #     VOLCANIC_ROCK_position = (
        #         state.position[Product.VOLCANIC_ROCK]
        #         if Product.VOLCANIC_ROCK in state.position
        #         else 0
        #     )

        #     K = self.params[option_product]['strike']
        #     T = self.params[option_product]['time to maturity']
        #     r = 0

        #     volatility = implied_volatility(
        #         option_mid_price,
        #         underlying_mid_price,
        #         self.params[option_product]["strike"],
        #         T,
        #         r
        #     )

        #     delta1 = delta(underlying_mid_price,K,T,r,volatility)

        #     option_orders, hedge_orders = self.hedge_option_orders_optimized(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

        #     result[Product.VOLCANIC_ROCK] = hedge_orders
        #     result[option_product] = option_orders

        #########################################################################################################################################################################
        
        ###################################### GAMMA and DELTA HEDGE ####################################################
        
        # option_product = Product.VOLCANIC_ROCK_VOUCHER_10500
        # underlying_prod = Product.VOLCANIC_ROCK
        # hedge_product = Product.VOLCANIC_ROCK_VOUCHER_10000

        # if option_product in self.params and state.order_depths[option_product].buy_orders.keys() and state.order_depths[option_product].sell_orders.keys() \
        #     and state.order_depths[underlying_prod].buy_orders.keys() and state.order_depths[underlying_prod].sell_orders.keys() \
        #     and state.order_depths[hedge_product].buy_orders.keys() and state.order_depths[hedge_product].sell_orders.keys():


        #     option_product_position = (
        #         state.position[option_product]
        #         if option_product in state.position
        #         else 0
        #     )
        #     option_orders = self.VolcanicRockCoupon_strat(state.order_depths,option_product,option_product_position)
        #     #option_orders = self.VolcanicRockCoupon_strat_Zscore(state.order_depths,option_product,VOLCANIC_ROCK_VOUCHER_10000_position)
            

        #     underlying_prod_position = (
        #         state.position[underlying_prod]
        #         if underlying_prod in state.position
        #         else 0
        #     )

        #     if len(option_orders) > 0:
                        
        #         hedge_option_orders, underlying_hedge_orders = self.hedge_with_delta_gamma(
        #                                                             underlying_prod, 
        #                                                             option_product,
        #                                                             option_orders,
        #                                                             hedge_product,
        #                                                             state.order_depths,
        #                                                             underlying_prod_position)
                
        #         result[underlying_prod] = underlying_hedge_orders
        #         result[option_product] = option_orders
        #         result[hedge_product] = hedge_option_orders

        #########################################################################################################################################################################

        ###################################### TRADE SPREADS ####################################################
        
        # higher_prod = Product.VOLCANIC_ROCK_VOUCHER_9750
        # lower_prod = Product.VOLCANIC_ROCK_VOUCHER_9500

        # if state.order_depths[higher_prod].buy_orders.keys() and state.order_depths[higher_prod].sell_orders.keys() \
        #    and state.order_depths[lower_prod].buy_orders.keys() and state.order_depths[lower_prod].sell_orders.keys():
            
        #     threshold = 1

        #     higher_prod_orders, lower_prod_orders = self.vertical_spread_pair_strat(
        #                                     state.order_depths,
        #                                     state.position,
        #                                     lower_prod,
        #                                     higher_prod,
        #                                     threshold)
            
        #     result[higher_prod] = higher_prod_orders
        #     result[lower_prod] = lower_prod_orders

        
        #########################################################################################################################################################################

        basket_position_1 = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread_orders_1 = self.spread_orders_1(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position_1,
            self.params[Product.SPREAD_1],
        )
        if spread_orders_1 != None:
            result[Product.CROISSANTS] = spread_orders_1[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders_1[Product.JAMS]
            result[Product.DJEMBES] = spread_orders_1[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders_1[Product.PICNIC_BASKET1]
        
        basket_position = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position,
            self.params[Product.SPREAD],
        )
        if spread_orders != None:
            if spread_orders_1 != None:
                result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS] + spread_orders_1[Product.CROISSANTS]
            else:
                result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            if spread_orders_1 != None:
                result[Product.JAMS] = spread_orders[Product.JAMS] + spread_orders_1[Product.JAMS]
            else:
                result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread_orders[Product.PICNIC_BASKET2]

        
        trader_data = json.dumps(traderObject, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
