from AlgorithmImports import *
from datetime import timedelta, time


class TopstepMNQEvalPassV2(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(50000)
        self.set_time_zone("America/Chicago")

        # ----------------------------
        # Core settings
        # ----------------------------
        self.bar_minutes = 5
        self.contract_qty = 1

        # Filters
        self.fast_len = 21
        self.slow_len = 50
        self.rsi_len = 14
        self.atr_len = 14
        self.vol_len = 20

        # Eval-focused controls
        self.max_trades_per_day = 1
        self.daily_loss_limit = -200
        self.daily_profit_lock = 300
        self.cooldown_bars = 12

        # Trade management
        self.atr_stop_mult = 1.0
        self.rr_target = 1.25
        self.be_trigger_r = 0.8
        self.time_stop_bars = 8

        # ----------------------------
        # Add MNQ continuous future
        # ----------------------------
        future = self.add_future(
            Futures.Indices.MICRO_NASDAQ_100_E_MINI,
            Resolution.MINUTE,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0
        )
        future.set_filter(0, 90)

        self.future = future
        self.continuous_symbol = future.symbol

        # ----------------------------
        # 5-minute consolidator
        # ----------------------------
        self.consolidator = TradeBarConsolidator(timedelta(minutes=self.bar_minutes))
        self.consolidator.data_consolidated += self.on_five_minute_bar
        self.subscription_manager.add_consolidator(self.continuous_symbol, self.consolidator)

        # ----------------------------
        # Indicators
        # ----------------------------
        self._fast_ema = ExponentialMovingAverage(self.fast_len)
        self._slow_ema = ExponentialMovingAverage(self.slow_len)
        self._rsi = RelativeStrengthIndex(self.rsi_len, MovingAverageType.WILDERS)
        self._atr = AverageTrueRange(self.atr_len, MovingAverageType.WILDERS)
        self._vol_sma = SimpleMovingAverage(self.vol_len)

        # ----------------------------
        # Session state
        # ----------------------------
        self.session_key = None
        self.day_start_value = 0.0
        self.daily_lockout = False
        self.trades_today = 0
        self.bar_index = 0
        self.last_exit_bar_index = -100000

        # Opening range
        self.or_high = None
        self.or_low = None
        self.or_ready = False

        # VWAP
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.current_vwap = None

        # Position state
        self.current_stop = None
        self.current_target = None
        self.current_trade_risk = None
        self.current_trade_entry_bar = None

        self.set_warm_up(timedelta(days=10))

    def on_data(self, slice: Slice):
        self._check_session_reset()

        # Mandatory flatten before Topstep cutoff
        if self._in_force_flat_window(self.time) and self.portfolio.invested:
            mapped = self.future.mapped
            if mapped is not None:
                self.liquidate(mapped)
            self.daily_lockout = True

    def on_five_minute_bar(self, sender, bar: TradeBar):
        self.bar_index += 1
        self._check_session_reset(bar.end_time)

        # Update indicators
        price_point = IndicatorDataPoint(bar.end_time, float(bar.close))
        vol_point = IndicatorDataPoint(bar.end_time, float(bar.volume))

        self._fast_ema.update(price_point)
        self._slow_ema.update(price_point)
        self._rsi.update(price_point)
        self._atr.update(bar)
        self._vol_sma.update(vol_point)

        # Reset/update VWAP
        if self._is_session_reset_bar(bar.end_time):
            self.vwap_num = 0.0
            self.vwap_den = 0.0

        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        self.vwap_num += typical * float(bar.volume)
        self.vwap_den += float(bar.volume)
        self.current_vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else None

        # Build opening range: 8:35, 8:40, 8:45 CT bars
        if self._in_opening_range_window(bar.end_time):
            self.or_high = float(bar.high) if self.or_high is None else max(self.or_high, float(bar.high))
            self.or_low = float(bar.low) if self.or_low is None else min(self.or_low, float(bar.low))

        if self._opening_range_complete(bar.end_time):
            if self.or_high is not None and self.or_low is not None:
                self.or_ready = True

        if self.is_warming_up:
            return

        if not self._indicators_ready():
            return

        self._manage_position(bar)

        # No new entries outside strict pass-eval conditions
        if self.daily_lockout:
            return
        if self.portfolio.invested:
            return
        if not self.or_ready:
            return
        if not self._in_entry_window(bar.end_time):
            return
        if self._in_force_flat_window(bar.end_time):
            return
        if self.trades_today >= self.max_trades_per_day:
            self.daily_lockout = True
            return
        if self.bar_index - self.last_exit_bar_index < self.cooldown_bars:
            return

        mapped = self.future.mapped
        if mapped is None or mapped not in self.securities:
            return

        min_tick = float(self.securities[mapped].symbol_properties.minimum_price_variation)
        or_width = self.or_high - self.or_low

        # Skip dead/erratic opens
        if self._atr.current.value == 0:
            return
        if or_width < float(self._atr.current.value) * 0.35:
            return
        if or_width > float(self._atr.current.value) * 1.75:
            return

        trend_up = (
            bar.close > self._slow_ema.current.value and
            self._fast_ema.current.value > self._slow_ema.current.value and
            self.current_vwap is not None and
            bar.close > self.current_vwap
        )

        trend_down = (
            bar.close < self._slow_ema.current.value and
            self._fast_ema.current.value < self._slow_ema.current.value and
            self.current_vwap is not None and
            bar.close < self.current_vwap
        )

        vol_ok = (
            self._vol_sma.current.value == 0 or
            float(bar.volume) >= self._vol_sma.current.value * 1.10
        )

        body = abs(float(bar.close) - float(bar.open))
        full_range = max(float(bar.high) - float(bar.low), min_tick)
        body_ratio = body / full_range

        long_signal = (
            trend_up and
            float(bar.close) > self.or_high and
            float(bar.close) > float(bar.open) and
            self._rsi.current.value >= 55 and
            body_ratio >= 0.55 and
            vol_ok
        )

        short_signal = (
            trend_down and
            float(bar.close) < self.or_low and
            float(bar.close) < float(bar.open) and
            self._rsi.current.value <= 45 and
            body_ratio >= 0.55 and
            vol_ok
        )

        if long_signal:
            raw_stop = min(self.or_low, float(bar.close) - float(self._atr.current.value) * self.atr_stop_mult)
            entry_risk = float(bar.close) - raw_stop

            if entry_risk > min_tick * 8:
                self.market_order(mapped, self.contract_qty, tag="MNQ ORB Long")
                self.current_stop = raw_stop
                self.current_target = float(bar.close) + entry_risk * self.rr_target
                self.current_trade_risk = entry_risk
                self.current_trade_entry_bar = self.bar_index
                self.trades_today += 1

        elif short_signal:
            raw_stop = max(self.or_high, float(bar.close) + float(self._atr.current.value) * self.atr_stop_mult)
            entry_risk = raw_stop - float(bar.close)

            if entry_risk > min_tick * 8:
                self.market_order(mapped, -self.contract_qty, tag="MNQ ORB Short")
                self.current_stop = raw_stop
                self.current_target = float(bar.close) - entry_risk * self.rr_target
                self.current_trade_risk = entry_risk
                self.current_trade_entry_bar = self.bar_index
                self.trades_today += 1

        self._update_daily_lockout()

    def _manage_position(self, bar: TradeBar):
        if not self.portfolio.invested:
            return

        mapped = self.future.mapped
        if mapped is None or mapped not in self.portfolio:
            return

        holdings = self.portfolio[mapped]
        if not holdings.invested:
            return

        high = float(bar.high)
        low = float(bar.low)
        avg_price = float(holdings.average_price)

        # Breakeven stop after partial progress
        if self.current_trade_risk is not None:
            if holdings.quantity > 0 and high >= avg_price + self.current_trade_risk * self.be_trigger_r:
                self.current_stop = max(self.current_stop, avg_price)

            if holdings.quantity < 0 and low <= avg_price - self.current_trade_risk * self.be_trigger_r:
                self.current_stop = min(self.current_stop, avg_price)

        exit_now = False

        if holdings.quantity > 0:
            if self.current_stop is not None and low <= self.current_stop:
                exit_now = True
            elif self.current_target is not None and high >= self.current_target:
                exit_now = True

        elif holdings.quantity < 0:
            if self.current_stop is not None and high >= self.current_stop:
                exit_now = True
            elif self.current_target is not None and low <= self.current_target:
                exit_now = True

        # Time stop
        if self.current_trade_entry_bar is not None:
            if self.bar_index - self.current_trade_entry_bar >= self.time_stop_bars:
                exit_now = True

        # Daily lockout / mandatory flatten
        self._update_daily_lockout()
        if self.daily_lockout or self._in_force_flat_window(bar.end_time):
            exit_now = True

        if exit_now:
            self.liquidate(mapped)
            self.last_exit_bar_index = self.bar_index
            self.current_stop = None
            self.current_target = None
            self.current_trade_risk = None
            self.current_trade_entry_bar = None

    def _update_daily_lockout(self):
        day_pnl = self.portfolio.total_portfolio_value - self.day_start_value

        if day_pnl <= self.daily_loss_limit:
            self.daily_lockout = True
        if day_pnl >= self.daily_profit_lock:
            self.daily_lockout = True
        if self.trades_today >= self.max_trades_per_day:
            self.daily_lockout = True

    def _check_session_reset(self, current_time=None):
        t = current_time if current_time is not None else self.time

        # Topstep-style session key: 5 PM CT boundary
        key = (t + timedelta(hours=7)).date()

        if self.session_key is None:
            self.session_key = key
            self.day_start_value = self.portfolio.total_portfolio_value
            return

        if key != self.session_key:
            self.session_key = key
            self.day_start_value = self.portfolio.total_portfolio_value
            self.daily_lockout = False
            self.trades_today = 0
            self.or_high = None
            self.or_low = None
            self.or_ready = False
            self.current_stop = None
            self.current_target = None
            self.current_trade_risk = None
            self.current_trade_entry_bar = None

    def _is_session_reset_bar(self, t):
        return t.hour == 17 and t.minute < self.bar_minutes

    def _in_opening_range_window(self, t):
        # Collect first 15 minutes after cash open on 5m bars
        return (
            (t.hour == 8 and t.minute >= 35) or
            (t.hour == 8 and t.minute == 40) or
            (t.hour == 8 and t.minute == 45)
        )

    def _opening_range_complete(self, t):
        return t.hour == 8 and t.minute >= 45

    def _in_entry_window(self, t):
        # Morning only
        after_open = (t.hour > 8 or (t.hour == 8 and t.minute >= 50))
        before_cut = (t.hour < 10 or (t.hour == 10 and t.minute <= 15))
        return after_open and before_cut

    def _in_force_flat_window(self, t):
        # Be flat before 3:10 PM CT
        return (t.hour == 15 and t.minute >= 5) or (t.hour > 15 and t.hour < 16)

    def _indicators_ready(self):
        return (
            self._fast_ema.is_ready and
            self._slow_ema.is_ready and
            self._rsi.is_ready and
            self._atr.is_ready and
            self._vol_sma.is_ready and
            self.current_vwap is not None
        )
