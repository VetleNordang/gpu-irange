import time
from datetime import datetime

# Create log file in the same directory
log_file = f"test_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

with open(log_file, 'w') as f:
    f.write(f"Starting test run at {datetime.now()}\n")
    f.flush()
    
    for i in range(361):
        f.write(f"{i}\n")
        f.flush()
        time.sleep(1)
    
    f.write("Finished counting to 360 seconds.\n")
    f.flush()

print(f"Log written to: {log_file}")