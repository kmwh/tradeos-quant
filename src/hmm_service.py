import datetime
import ccxt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
from config import HMM_CONFIG

def calculate_realtime_hmm_score(symbol: str) -> int:
    exchange = ccxt.binance({'enableRateLimit': True})
    formatted_symbol = symbol.replace("/", "") if "/" in symbol else symbol
    
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    start_ts = int((now_utc - datetime.timedelta(days=HMM_CONFIG['FETCH_DAYS'])).timestamp() * 1000)
    end_ts = int(now_utc.timestamp() * 1000)
    
    all_ohlcv = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(formatted_symbol, HMM_CONFIG['TIMEFRAME'], since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"CCXT Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    price_change_abs = df['close'].diff(HMM_CONFIG['ER_WINDOW']).abs()
    volatility_sum = df['close'].diff(1).abs().rolling(HMM_CONFIG['ER_WINDOW']).sum()
    df['abs_er'] = price_change_abs / (volatility_sum + 1e-8)
    
    factor = 1.0 / (4.0 * np.log(2.0))
    hl_log_sq = (np.log(df['high'] / df['low'].replace(0, np.nan))) ** 2
    df['log_parkinson'] = np.log(np.sqrt(factor * hl_log_sq) + 1e-8)
    df['rvol'] = df['volume'] / (df['volume'].rolling(window=HMM_CONFIG['RVOL_WINDOW']).mean() + 1e-8)
    
    df = df.dropna()
    features = ['abs_er', 'log_parkinson', 'rvol']
    
    last_date = df.index[-1]
    
    test_start = last_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    train_end = test_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=HMM_CONFIG['TRAIN_WINDOW'])
    
    train_df = df.loc[train_start:train_end]
    
    context_start = last_date - pd.Timedelta(days=HMM_CONFIG['CONTEXT_WINDOW'])
    context_df = df.loc[context_start:last_date]
    
    if len(train_df) < HMM_CONFIG['TRAIN_WINDOW'] * 0.8 or len(context_df) < 5:
        return 50 
        
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_train = np.clip(X_train, -3.0, 3.0)
    
    hmm = GaussianHMM(n_components=HMM_CONFIG['HMM_STATES'], covariance_type="full", n_iter=200, random_state=42)
    hmm.fit(X_train)
    
    state_means = {i: hmm.means_[i][0] for i in range(HMM_CONFIG['HMM_STATES'])}
    sorted_states = sorted(state_means.keys(), key=lambda k: state_means[k])
    
    trend_idx = sorted_states[1] 
    
    eval_scaled = scaler.transform(context_df[features])
    eval_scaled = np.clip(eval_scaled, -3.0, 3.0)
    
    step_probas = hmm.predict_proba(eval_scaled)[-1] 
    prob = step_probas[trend_idx]
    
    return int(np.round(prob * 100))