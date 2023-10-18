from datetime import datetime


# Get the current UTC time
current_time = datetime.utcnow().time()

# Define the trading sessions in UTC time
sydney_session_start = datetime.strptime('22:00:00', '%H:%M:%S').time()
sydney_session_end = datetime.strptime('07:00:00', '%H:%M:%S').time()
tokyo_session_start = datetime.strptime('00:00:00', '%H:%M:%S').time()
tokyo_session_end = datetime.strptime('09:00:00', '%H:%M:%S').time()
london_session_start = datetime.strptime('08:00:00', '%H:%M:%S').time()
london_session_end = datetime.strptime('17:00:00', '%H:%M:%S').time()
newyork_session_start = datetime.strptime('13:00:00', '%H:%M:%S').time()
newyork_session_end = datetime.strptime('22:00:00', '%H:%M:%S').time()

# Check if the current time falls within any trading session
is_forex_market_open = (
    (current_time >= sydney_session_start and current_time <= sydney_session_end) or
    (current_time >= tokyo_session_start and current_time <= tokyo_session_end) or
    (current_time >= london_session_start and current_time <= london_session_end) or
    (current_time >= newyork_session_start and current_time <= newyork_session_end)
)

if is_forex_market_open:
    print("Forex market is currently open.", current_time)
else:
    print("Forex market is currently closed.")
