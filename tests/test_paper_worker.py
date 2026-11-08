import os
import tempfile
import unittest
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from execution_engine import TZ
from paper_worker import (
    PaperWorker, calculate_atr, completed_daily_bars, parse_watchlist,
    quote_is_fresh, rank_entry_candidates, round_price_down, max_shares_for_participation,
)


class PaperWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trading.sqlite3")
        self.market_db = os.path.join(self.tmp.name, "market.sqlite3")
        self.now = datetime(2026, 11, 9, 10, 0, tzinfo=TZ)

    def tearDown(self):
        self.tmp.cleanup()

    def test_watchlist_parser_and_limit(self):
        self.assertEqual(
            parse_watchlist("2330:上市,6488:上櫃"),
            {"2330": "上市", "6488": "上櫃"},
        )
        with self.assertRaises(ValueError):
            parse_watchlist("2330")
        with self.assertRaises(ValueError):
            parse_watchlist("2330:上市,2330:上櫃")
        with self.assertRaises(ValueError):
            parse_watchlist(",".join(f"{1000+i:04d}:上市" for i in range(31)))

    def test_quote_freshness_rejects_stale_and_future_data(self):
        fresh = {"price": 100, "timestamp": self.now.isoformat(), "source": "test"}
        self.assertTrue(quote_is_fresh(fresh, self.now))
        stale = dict(fresh, timestamp=(self.now - timedelta(seconds=61)).isoformat())
        self.assertFalse(quote_is_fresh(stale, self.now))
        future = dict(fresh, timestamp=(self.now + timedelta(seconds=10)).isoformat())
        self.assertFalse(quote_is_fresh(future, self.now))

    def test_price_rounding_uses_taiwan_tick_size(self):
        self.assertEqual(round_price_down(100.49), 100.0)
        self.assertEqual(round_price_down(49.99), 49.95)
        self.assertEqual(round_price_down(49.99, "etf"), 49.99)

    def test_completed_daily_bars_excludes_today_partial_bar(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-11-06", "2026-11-09"]),
            "open": [10, 11], "high": [11, 12], "low": [9, 10],
            "close": [10, 11], "volume": [100, 100],
        })
        result = completed_daily_bars(frame, self.now.date())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[-1]["date"].date().isoformat(), "2026-11-06")

    def test_order_size_respects_prior_completed_average_turnover(self):
        frame = pd.DataFrame({
            "close": np.full(20, 100.0),
            "volume": np.full(20, 1000),
        })
        self.assertEqual(max_shares_for_participation(frame, 100.0, 0.01), 10)
        self.assertEqual(max_shares_for_participation(frame.iloc[:19], 100.0, 0.01), 0)
        self.assertEqual(max_shares_for_participation(frame, 100.0, 0.02), 20)

    def test_invalid_participation_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "參與率"):
            max_shares_for_participation(
                pd.DataFrame({"close": [10] * 20, "volume": [1000] * 20}),
                10, 0.5,
            )

    def test_atr_uses_true_range(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=20),
            "high": np.arange(20) + 12.0,
            "low": np.arange(20) + 10.0,
            "close": np.arange(20) + 11.0,
        })
        self.assertGreater(calculate_atr(frame), 0)

    def test_worker_applies_current_day_corporate_action_before_matching(self):
        from unittest.mock import patch
        worker = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={"2330":"上市"}, auto_entries=False,
            now_provider=lambda: self.now,
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO paper_positions(symbol,quantity,average_cost,position_version) VALUES(?,?,?,?)",
                ("2330",1000,10.0,1),
            )
            conn.execute(
                "INSERT INTO paper_fills(fill_id,order_id,symbol,side,quantity,price,gross,fees,tax,realized_pnl,quote_source,quote_time,filled_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("f1","o1","2330","BUY",1000,10.0,10000.0,14.25,0.0,0.0,
                 "test",self.now.isoformat(),self.now.isoformat()),
            )
            conn.execute("UPDATE paper_account SET cash=89985.75 WHERE id=1")
        events = pd.DataFrame([{
            "symbol":"2330","effective_date":self.now.date().isoformat(),
            "action_type":"stock_split","share_multiplier":2.0,
            "cash_per_share":0.0,"payment_date":None,
            "reference_price":5.0,"event_id":"worker-split",
        }])
        with patch.object(worker, "_load_corporate_actions", return_value=events):
            result = worker._process_corporate_actions(self.now)
        self.assertEqual(result["applied"], ["worker-split"])
        self.assertEqual(worker.account.snapshot()["positions"][0]["quantity"], 2000)

    def test_worker_blocks_when_past_corporate_action_was_missed(self):
        from unittest.mock import patch
        worker = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={"2330":"上市"}, auto_entries=False,
            now_provider=lambda: self.now,
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO paper_positions(symbol,quantity,average_cost,position_version) VALUES(?,?,?,?)",
                ("2330",1000,10.0,1),
            )
            conn.execute(
                "INSERT INTO paper_fills(fill_id,order_id,symbol,side,quantity,price,gross,fees,tax,realized_pnl,quote_source,quote_time,filled_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("f1","o1","2330","BUY",1000,10.0,10000.0,14.25,0.0,0.0,
                 "test",self.now.isoformat(),self.now.isoformat()),
            )
            conn.execute("UPDATE paper_account SET cash=89985.75 WHERE id=1")
        events = pd.DataFrame([{
            "symbol":"2330","effective_date":"2026-11-08",
            "action_type":"stock_split","share_multiplier":2.0,
            "cash_per_share":0.0,"payment_date":None,
            "reference_price":5.0,"event_id":"worker-missed",
        }])
        with patch.object(worker, "_load_corporate_actions", return_value=events):
            with self.assertRaisesRegex(RuntimeError, "未處理的過去除權息事件"):
                worker._process_corporate_actions(self.now)
        self.assertEqual(worker.account.snapshot()["positions"][0]["quantity"], 1000)

    def test_worker_lease_prevents_two_processes(self):
        worker_a = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={}, auto_entries=False,
        )
        worker_b = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={}, auto_entries=False,
        )
        self.assertTrue(worker_a.acquire_lease(self.now))
        self.assertFalse(worker_b.acquire_lease(self.now))
        worker_a.release_lease()
        self.assertTrue(worker_b.acquire_lease(self.now))
        worker_b.release_lease()

    def test_worker_blocks_all_trading_when_account_ledger_is_corrupt(self):
        import sqlite3
        worker = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={"2330": "上市"}, auto_entries=True,
            now_provider=lambda: self.now,
            quote_fetcher=lambda symbol, market: self.fail("corrupt ledger must block quote requests"),
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE paper_account SET cash=cash-1 WHERE id=1")
        result = worker.run_once(self.now)
        self.assertEqual(result["status"], "帳本完整性異常，已停止交易")
        self.assertFalse(result["ledger_integrity"]["ok"])
        self.assertEqual(result["entries"], 0)
        self.assertEqual(result["fills"], 0)

    def test_worker_skips_stale_quotes_without_loading_history(self):
        worker = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={"2330": "上市"}, auto_entries=True,
            now_provider=lambda: self.now,
            calendar_checker=lambda day, market: (True, "test calendar"),
            quote_fetcher=lambda symbol, market: {
                "price": 100.0,
                "timestamp": (self.now - timedelta(seconds=61)).isoformat(),
                "source": "test",
            },
            history_loader=lambda symbol, market, now: self.fail("stale quote must stop before history scan"),
        )
        result = worker.run_once(self.now)
        self.assertEqual(result["status"], "沒有可用的新鮮報價")
        self.assertEqual(result["entries"], 0)

    def test_empty_watchlist_fails_closed_without_market_data(self):
        worker = PaperWorker(
            db_path=self.db, market_db_path=self.market_db,
            watchlist={}, auto_entries=True,
            now_provider=lambda: self.now,
        )
        result = worker.run_once(self.now)
        self.assertEqual(result["status"], "等待設定 PAPER_WATCHLIST")
        self.assertEqual(result["entries"], 0)


if __name__ == "__main__":
    unittest.main()
