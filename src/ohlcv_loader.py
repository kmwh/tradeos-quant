import datetime
import ccxt
from config import HMM_CONFIG

def load_ohlcv(symbol: str) -> list:
  exchange = ccxt.binance({'enableRateLimit': True})
  
  now_utc = datetime.datetime.now(datetime.timezone.utc)
  start_ts = int((now_utc - datetime.timedelta(days=HMM_CONFIG['FETCH_DAYS'])).timestamp() * 1000)
  end_ts = int(now_utc.timestamp() * 1000)
  
  all_ohlcv = []
  current_ts = start_ts
  
  while current_ts < end_ts:
      try:
          ohlcv = exchange.fetch_ohlcv(symbol, HMM_CONFIG['TIMEFRAME'], since=current_ts, limit=1000)
          if not ohlcv:
              break
          all_ohlcv.extend(ohlcv)
          current_ts = ohlcv[-1][0] + 1
          if len(ohlcv) < 1000:
              break
      except Exception as e:
          print(f"CCXT Fetch Error: {e}")
          break
  
  return all_ohlcv