from __future__ import annotations

import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, raw_df: pd.DataFrame | None = None, initial_balance: float = 10_000, transaction_fee_percent: float = 0.001):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("DataFrame `df` must be non-empty.")

        self.df = df.reset_index(drop=True)
        self.raw_df = raw_df.reset_index(drop=True) if raw_df is not None else self.df

        if len(self.raw_df) != len(self.df):
            raise ValueError("`raw_df` and `df` must have the same number of rows.")

        self.initial_balance = float(initial_balance)
        self.transaction_fee_percent = float(transaction_fee_percent)

        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.current_step = 0

        self.action_space = spaces.Discrete(3)

        self.feature_columns = list(self.df.columns)
        obs_dim = len(self.feature_columns) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_current_price(self) -> float:
        row = self.raw_df.iloc[self.current_step]
        if "Close" in self.raw_df.columns:
            return float(row["Close"])
        numeric_row = pd.to_numeric(row, errors="coerce").dropna()
        if numeric_row.empty:
            raise ValueError("No numeric price found in raw_df row.")
        return float(numeric_row.iloc[0])

    def _get_observation(self) -> np.ndarray:
        market_features = (
            pd.to_numeric(self.df.iloc[self.current_step], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        portfolio_state = np.array(
            [self.current_balance, self.shares_held, self.net_worth],
            dtype=np.float32,
        )
        return np.concatenate([market_features, portfolio_state])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        prev_net_worth = self.net_worth
        current_price = self._get_current_price()

        if action == 1:
            if self.current_balance > 0 and current_price > 0:
                shares_to_buy = self.current_balance / (current_price * (1.0 + self.transaction_fee_percent))
                fee = shares_to_buy * current_price * self.transaction_fee_percent
                self.shares_held += shares_to_buy
                self.current_balance -= shares_to_buy * current_price + fee

        elif action == 2:
            if self.shares_held > 0 and current_price > 0:
                gross = self.shares_held * current_price
                fee = gross * self.transaction_fee_percent
                self.current_balance += gross - fee
                self.shares_held = 0.0

        self.net_worth = self.current_balance + self.shares_held * current_price
        reward = float((self.net_worth - prev_net_worth) / self.initial_balance)

        self.current_step += 1
        reached_end = self.current_step >= len(self.df)
        terminated = reached_end or self.net_worth <= 0
        truncated = False

        if reached_end:
            self.current_step = len(self.df) - 1

        info = {
            "current_balance": self.current_balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
            "current_step": self.current_step,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step} | Balance: {self.current_balance:.2f} | "
            f"Shares: {self.shares_held:.6f} | Net Worth: {self.net_worth:.2f}"
        )
