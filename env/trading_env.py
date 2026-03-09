from __future__ import annotations

import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gymnasium.Env):
    """A simple single-asset trading environment with discrete actions.

    Actions:
    - 0: Hold
    - 1: Buy
    - 2: Sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("DataFrame `df` must be non-empty.")

        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.transaction_fee_percent = float(transaction_fee_percent)

        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.current_step = 0

        # 0 = Hold, 1 = Buy, 2 = Sell.
        self.action_space = spaces.Discrete(3)

        # Observation = market row features + [current_balance, shares_held, net_worth].
        self.feature_columns = list(self.df.columns)
        obs_dim = len(self.feature_columns) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_current_price(self):
        row = self.df.iloc[self.current_step]
        if "Close" in self.df.columns:
            return float(row["Close"])

        numeric_row = pd.to_numeric(row, errors="coerce")
        valid = numeric_row.dropna()
        if valid.empty:
            raise ValueError("No numeric price value available in current DataFrame row.")
        return float(valid.iloc[0])

    def _get_observation(self):
        """Build observation vector from market features and portfolio state."""
        row = self.df.iloc[self.current_step]
        market_features = pd.to_numeric(row, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        portfolio_state = np.array(
            [self.current_balance, self.shares_held, self.net_worth],
            dtype=np.float32,
        )
        return np.concatenate([market_features, portfolio_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.current_step = 0

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        prev_net_worth = self.net_worth
        current_price = self._get_current_price()

        # Execute action.
        if action == 1:
            if self.current_balance > 0 and current_price > 0:
                affordable_shares = self.current_balance / current_price
                fee = (affordable_shares * current_price) * self.transaction_fee_percent
                total_cost = (affordable_shares * current_price) + fee

                if total_cost > self.current_balance:
                    affordable_shares = self.current_balance / (current_price * (1 + self.transaction_fee_percent))
                    fee = (affordable_shares * current_price) * self.transaction_fee_percent
                    total_cost = (affordable_shares * current_price) + fee

                self.shares_held += affordable_shares
                self.current_balance -= total_cost

        elif action == 2:  # Sell
            if self.shares_held > 0 and current_price > 0:
                gross_sale_value = self.shares_held * current_price
                fee = gross_sale_value * self.transaction_fee_percent
                net_sale_value = gross_sale_value - fee

                self.current_balance += net_sale_value
                self.shares_held = 0.0

        # Hold
        self.net_worth = self.current_balance + (self.shares_held * current_price)
        reward = self.net_worth - prev_net_worth

        self.current_step += 1

        reached_end = self.current_step >= len(self.df)
        bankrupt = self.net_worth < 0
        terminated = bankrupt or reached_end
        truncated = False

        if reached_end:
            self.current_step = len(self.df) - 1

        observation = self._get_observation()
        info = {
            "current_balance": self.current_balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
            "current_step": self.current_step,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, Balance: {self.current_balance:.2f}, "
            f"Shares: {self.shares_held:.6f}, Net Worth: {self.net_worth:.2f}"
        )
