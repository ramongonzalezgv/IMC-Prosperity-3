
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import jsonpickle
import numpy as np
import math
import json




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
        "reversion_beta": -0.229,
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
            Product.PICNIC_BASKET2: 100
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
    
    def construct_artificial_basket1_order_depth(
        self,
        order_depths
        ):
        
        # Initialize object where we will store the artificial basket order depths
        artificial_basket1_order_depths = OrderDepth()

        # Weights of the products
        croissants_weight = BASKET1_PRODS[Product.CROISSANTS]
        jams_weight = BASKET1_PRODS[Product.JAMS]
        djembes_weight = BASKET1_PRODS[Product.DJEMBES]

        # Get best bid and ask for each of the products of the artificial basket
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float('inf')
        
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')

        djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
        djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float('inf')

        # Get implied bid and ask for the basket
        if (croissants_best_bid != 0) & (jams_best_bid != 0) & (djembes_best_bid != 0): # If you don't have at least one bid for each of the products you can't make the basket
            artificial_basket1_implied_bid = croissants_best_bid*croissants_weight + jams_best_bid*jams_weight + djembes_best_bid*djembes_weight
        else:
            artificial_basket1_implied_bid = 0

        # If there is one or more asks missing, the implied ask will be inf
        artificial_basket1_implied_ask = croissants_best_ask*croissants_weight + jams_best_ask*jams_weight + djembes_best_ask*djembes_weight

        # Get maximum number of artificial baskets you could make at those price levels (artificial basket bid volume and ask volume)
        if artificial_basket1_implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid] // croissants_weight
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // jams_weight
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // djembes_weight

            artificial_basket1_bid_volume = min(croissants_bid_volume,jams_bid_volume,djembes_bid_volume)
            artificial_basket1_order_depths.buy_orders[artificial_basket1_implied_bid] = artificial_basket1_bid_volume
        
        if artificial_basket1_implied_ask < float('inf'):
            croissants_ask_volume = order_depths[Product.CROISSANTS].buy_orders[croissants_best_ask] // croissants_weight
            jams_ask_volume = order_depths[Product.JAMS].buy_orders[jams_best_ask] // jams_weight
            djembes_ask_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_ask] // djembes_weight

            artificial_basket1_ask_volume = min(croissants_ask_volume,jams_ask_volume,djembes_ask_volume)
            artificial_basket1_order_depths.buy_orders[artificial_basket1_implied_ask] = -artificial_basket1_ask_volume

        return artificial_basket1_order_depths
    
    def construct_artificial_basket_order_depth( # generalized version of the previous function
        self, 
        order_depths: Dict[str, OrderDepth],
        baskets_products: Dict[Product, Dict[Product, int]], # Dictionary with the products that the basket contains and their weights
        basket : Product
        ):

        basket_prods = baskets_products[basket]

        # Initialize object where we will store the artificial basket order depths
        artificial_basket_order_depths = OrderDepth()
        
        # Initialize variables to store the best bid, ask, and volumes
        best_bids = {}
        best_asks = {}
        bid_volumes = {}
        ask_volumes = {}
        
        for product, weight in basket_prods.items():
            # Get best bid and ask for each product
            best_bid = max(order_depths[product].buy_orders.keys()) if order_depths[product].buy_orders else 0
            best_ask = min(order_depths[product].sell_orders.keys()) if order_depths[product].sell_orders else float('inf')
            
            best_bids[product] = best_bid
            best_asks[product] = best_ask
            
            # Get the volumes for bid and ask if applicable
            if best_bid > 0:
                bid_volumes[product] = order_depths[product].buy_orders[best_bid] // weight
            if best_ask < float('inf'):
                ask_volumes[product] = order_depths[product].sell_orders[best_ask] // weight
        
        # Calculate the implied bid and ask for the artificial basket
        if all(best_bids[p] != 0 for p in basket_prods):
            artificial_basket_implied_bid = sum(best_bids[p] * basket_prods[p] for p in basket_prods)
        else:
            artificial_basket_implied_bid = 0

        artificial_basket_implied_ask = sum(best_asks[p] * basket_prods[p] for p in basket_prods)
        
        # Determine the maximum number of artificial baskets at those price levels
        if artificial_basket_implied_bid > 0:
            artificial_basket_bid_volume = min(bid_volumes.values())
            artificial_basket_order_depths.buy_orders[artificial_basket_implied_bid] = artificial_basket_bid_volume

        if artificial_basket_implied_ask < float('inf'):
            artificial_basket_ask_volume = min(ask_volumes.values())
            artificial_basket_order_depths.sell_orders[artificial_basket_implied_ask] = -artificial_basket_ask_volume

        return artificial_basket_order_depths
    
    def convert_artificial_basket_orders(self, 
        artificial_orders: List[Order],
        order_depths: Dict[str, OrderDepth],
        baskets_products: Dict[Product, Dict[Product, int]],
        basket: product
    ) -> Dict[str, List[Order]]:
        """
        Convert orders on the artificial basket into orders on the underlying components.
        
        Args:
            artificial_orders: List of orders executed on the artificial basket.
            order_depths: Dictionary mapping products to their current OrderDepth.
            baskets_products: Dictionary of baskets and their underlying composition & weights.
            basket: The basket identifier (e.g. Product.PICNIC_BASKET1) to process.
        
        Returns:
            A dictionary mapping each underlying product (in the basket) to a list of Orders.
        """
        # Retrieve the composition (e.g. {CROISSANTS:6, JAMS:3, DJEMBES:1})
        basket_composition = baskets_products[basket]
        
        # Initialize dictionary: each underlying will have its own list of orders
        component_orders = {product: [] for product in basket_composition.keys()}
        
        # Build the current artificial basket order depth using our generalized function.
        artificial_order_depth = self.construct_artificial_basket_order_depth(order_depths, baskets_products, basket)
        
        # Identify best bid and ask on the artificial basket order depth.
        best_bid = max(artificial_order_depth.buy_orders.keys()) if artificial_order_depth.buy_orders else 0
        best_ask = min(artificial_order_depth.sell_orders.keys()) if artificial_order_depth.sell_orders else float('inf')
        
        # Iterate through each artificial basket order that we wish to convert.
        for order in artificial_orders:
            price = order.price
            quantity = order.quantity
            
            # Checks that the price generated by the trade signal is aligned with the market's current pricing for the artificial basket constructed
            if quantity > 0 and price >= best_ask:
                # Buy order on the artificial basket: implies buying underlying at their best ask prices.
                for product, weight in basket_composition.items():
                    # The best sell (ask) available for the product.
                    underlying_price = min(order_depths[product].sell_orders.keys())
                    component_orders[product].append(Order(product, underlying_price, quantity * weight))
            elif quantity < 0 and price <= best_bid:
                # Sell order on the artificial basket: implies selling underlying at their best bid prices.
                for product, weight in basket_composition.items():
                    # The best buy (bid) available for the product.
                    underlying_price = max(order_depths[product].buy_orders.keys())
                    component_orders[product].append(Order(product, underlying_price, quantity * weight))
            else:
                # Order does not meet criteria for conversion.
                continue
        
        return component_orders

    def spread_orders(self, 
        order_depths: Dict[str, OrderDepth],
        basket,
        basket_position: int,
        spread_history: List[float],
        baskets_products: Dict[Product, Dict[Product, int]]
    ):
        """
        Monitor the spread between the actual basket order book and the artificial basket
        (constructed from component orders). When the spread (measured in a z-score)
        exceeds a predefined threshold, prepare and return orders for arbitrage.
        
        Args:
            order_depths: Dictionary of OrderDepth for each product.
            basket: The basket identifier (e.g. Product.PICNIC_BASKET1).
            basket_position: The current net position for the actual basket.
            spread_history: Running list of recent spread values.
            baskets_products: Dictionary containing compositions for all baskets.
        
        Returns:
            A dictionary mapping product identifiers to lists of Orders if an arbitrage opportunity exists;
            Otherwise, returns None.
        """
        # Get the order depth for the actual basket.
        basket_order_depth = order_depths[basket]
        
        # Build the artificial basket order depth from component order books.
        artificial_order_depth = self.construct_artificial_basket_order_depth(order_depths, baskets_products, basket)
        
        # Compute the mid-price for each: typically (best bid + best ask)/2.
        basket_swmid = self.get_swmid(basket_order_depth)
        artificial_swmid = self.get_swmid(artificial_order_depth)
        
        # Calculate the spread between the actual basket and the artificial one.
        spread = basket_swmid - artificial_swmid
        spread_history.append(spread)
        
        # Ensure we have sufficient history to compute meaningful statistics.
        window = self.params[basket]["spread_std_window"]
        if len(spread_history) < window:
            return None
        
        # Compute spread mean and standard deviation (here we incorporate starting iterations if needed).
        spread_mean = (np.sum(spread_history) + (self.params[basket]["spread_mean"] * self.params[basket]["starting_its"])) \
                    / (self.params[basket]["starting_its"] + len(spread_history))
        spread_std = np.std(spread_history[-window:])
        zscore = (spread - spread_mean) / spread_std
        
        # Decide on action based on the z-score thresholds.
        if zscore >= self.params[basket]["zscore_threshold"]:
            # The basket is trading at a premium relative to its artificial value.
            if basket_position == -self.params[basket]["target_position"]:
                return None
            target_quantity = abs(-self.params[basket]["target_position"] - basket_position)
            
            # Use available volumes from the order book for the basket and artificial basket.
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            artificial_ask_price = min(artificial_order_depth.sell_orders.keys())
            artificial_ask_volume = abs(artificial_order_depth.sell_orders[artificial_ask_price])
            
            orderbook_volume = min(basket_bid_volume, artificial_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            # Prepare the actual basket order (sell the basket).
            basket_orders = [Order(basket, basket_bid_price, -execute_volume)]
            
            # Prepare an artificial basket order (buy the artificial basket).
            # Here we label the order with a synthetic identifier (for example, prefixing with "ARTIFICIAL_").
            artificial_orders = [Order("ARTIFICIAL_" + basket, artificial_ask_price, execute_volume)]
            
            # Convert the artificial basket order into underlying orders.
            aggregate_orders = self.convert_artificial_basket_orders(artificial_orders, order_depths, baskets_products, basket)
            aggregate_orders[basket] = basket_orders
            return aggregate_orders
        
        if zscore <= -self.params[basket]["zscore_threshold"]:
            # The basket is trading at a discount relative to its artificial value.
            if basket_position == self.params[basket]["target_position"]:
                return None
            target_quantity = abs(self.params[basket]["target_position"] - basket_position)
            
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            artificial_bid_price = max(artificial_order_depth.buy_orders.keys())
            artificial_bid_volume = abs(artificial_order_depth.buy_orders[artificial_bid_price])
            
            orderbook_volume = min(basket_ask_volume, artificial_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            # Prepare the actual basket order (buy the basket).
            basket_orders = [Order(basket, basket_ask_price, execute_volume)]
            
            # Prepare an artificial basket order (sell the artificial basket).
            artificial_orders = [Order("ARTIFICIAL_" + basket, artificial_bid_price, -execute_volume)]
            
            # Convert the artificial basket order into individual component orders.
            aggregate_orders = self.convert_artificial_basket_orders(artificial_orders, order_depths, baskets_products, basket)
            aggregate_orders[basket] = basket_orders
            return aggregate_orders
        
        return None        



    def run(self, state: TradingState):
    
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

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
