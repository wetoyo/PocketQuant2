import os
import glob

log_files = glob.glob('execution/logs/fetch_data_*.log')
latest_log = max(log_files, key=os.path.getctime)

print(f"Checking log file: {latest_log}")
with open(latest_log, 'r') as f:
    content = f.read()
    if "Starting data fetch process..." in content and "Data fetch process completed successfully." in content:
        print("Log verification SUCCESS")
    else:
        print("Log verification FAILED")
        print("Content snippet:")
        print(content[:500])
