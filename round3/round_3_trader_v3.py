
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import jsonpickle
import numpy as np
from statistics import NormalDist
import math
import json

Dict = dict



def BS_CALL(S, K, T, r, sig):
    N = NormalDist().cdf
    print(S)
    print(K)
    print(S/K)
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
        Option_price, S, K, T, r, max_iterations=2000, tolerance=1e-10
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
    ARTIFICIAL_PICNIC_BASKET1 = "ARTIFICIAL_PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SPREAD1 = "SPREAD1"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    



BASKETS_PRODUCTS = {
    Product.PICNIC_BASKET1: 
    {
        Product.CROISSANTS : 6,
        Product.JAMS : 3,
        Product.DJEMBES : 1
    },
    Product.PICNIC_BASKET2:
    {
        Product.CROISSANTS : 4,
        Product.JAMS : 2
    }
}

BASKET1_PRODS = {
    Product.CROISSANTS : 6,
    Product.JAMS : 3,
    Product.DJEMBES : 1
}

BASKET2_PRODS = {
    Product.CROISSANTS : 4,
    Product.JAMS : 2
}


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
    Product.SPREAD1:{
        "spread_mean": 48.762433333333334,
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11,
        "target_position": 60,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.45,
        "threshold": 0.00163,
        "strike": 9500,
        "time to maturity": 5 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 9750,
        "time to maturity": 5 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 10000,
        "time to maturity": 5 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10250,
        "time to maturity": 5 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
     Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10500,
        "time to maturity": 5 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
}




class Trader:
    def __init__(self, params=None, baskets_products = None):
        if params is None:
            params = PARAMS
        self.params = params

        if baskets_products is None:
            baskets_products = BASKETS_PRODUCTS
        self.baskets_products = baskets_products

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
        
        if Product.VOLCANIC_ROCK not in order_depths.keys() or len(order_depths[Product.VOLCANIC_ROCK].buy_orders) == 0 or len(order_depths[Product.VOLCANIC_ROCK].sell_orders) == 0:
            return option_orders

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

        if self.LIMIT[prod] >= position:
            if BS_price > option_best_ask_price:
                quantity = min(option_best_ask_volume, self.LIMIT[prod] - position)  
                option_orders.append(Order(prod,option_best_ask_price,quantity))

            elif BS_price < option_best_bid_price:
                quantity = min(option_best_bid_volume, self.LIMIT[prod] - position)
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
        
        orders: List[Order] = []
        
        if option_orders == None or len(option_orders) == 0:
            option_position_after_trade = option_position
        else:
            option_position_after_trade = option_position + sum(order.quantity for order in option_orders)
        
        target_coconut_position = -delta * option_position_after_trade
        
        if target_coconut_position == prod_position:
            return orders
        
        target_coconut_quantity = target_coconut_position - prod_position

        
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(order_depths[underlying_prod].sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[underlying_prod] - prod_position,
            )
            if quantity > 0:
                orders.append(Order(underlying_prod, best_ask, round(quantity)))
        
        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(order_depths[underlying_prod].buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[underlying_prod] + prod_position,
            )
            if quantity > 0:
                orders.append(Order(underlying_prod, best_bid, -round(quantity)))
        
        return orders

        
        


    def run(self, state: TradingState):

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
    
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
        trader_data = ""

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            amethyst_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    amethyst_position,
                )
            )
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            amethyst_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
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
            result[Product.SQUID_INK] = (
                SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
            )

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

        if Product.VOLCANIC_ROCK_VOUCHER_10000 in self.params and Product.VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths.keys() and Product.VOLCANIC_ROCK in state.order_depths.keys():

            underlying_mid_price = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
            #underlying_mid_price = 1000
            option_mid_price = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000])

            VOLCANIC_ROCK_VOUCHER_10000_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_10000]
                if Product.VOLCANIC_ROCK_VOUCHER_10000 in state.position
                else 0
            )
            option_orders = self.VolcanicRockCoupon_strat(state.order_depths,Product.VOLCANIC_ROCK_VOUCHER_10000,VOLCANIC_ROCK_VOUCHER_10000_position)
            result[Product.VOLCANIC_ROCK_VOUCHER_10000] = option_orders

            VOLCANIC_ROCK_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )


            K = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['strike']
            T = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['time to maturity']
            r = 0

            volatility = implied_volatility(
                option_mid_price,
                underlying_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                T,
                r
            )

            delta1 = delta(underlying_mid_price,K,T,r,volatility)

            hedge_orders = self.hedge_option_orders(Product.VOLCANIC_ROCK,state.order_depths,option_orders,VOLCANIC_ROCK_position, VOLCANIC_ROCK_VOUCHER_10000_position,delta1)

            result[Product.VOLCANIC_ROCK] = hedge_orders

        # if Product.PICNIC_BASKET2 and Product.PICNIC_BASKET2 in state.order_depths:

        #     position = state.position[Product.PICNIC_BASKET2] if Product.VOLCANIC_ROCK_VOUCHER_9500 in state.position else 0

        #     basket2_orders = self.dummy(state.order_depths, Product.PICNIC_BASKET2,position)

        #     result[Product.PICNIC_BASKET2] = basket2_orders

        
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
