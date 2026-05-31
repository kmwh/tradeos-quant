from dotenv import load_dotenv

load_dotenv()

HMM_CONFIG = {
    'TIMEFRAME': '4h',             
    'ER_WINDOW': 84,               
    'RVOL_WINDOW': 120,            
    'HMM_STATES': 2,               
    'TRAIN_WINDOW': 365,           
    'CONTEXT_WINDOW': 30,          
    'FETCH_DAYS': 450 
}