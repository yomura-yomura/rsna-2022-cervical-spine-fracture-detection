import kaggle
import time
import datetime as dt
import sys


competition_name = "rsna-2022-cervical-spine-fracture-detection"
# username_to_be_monitored = "Ranchantan"
username_to_be_monitored = None

api = kaggle.KaggleApi()
api.authenticate()

refs = [
    sub.ref
    for sub in api.competition_submissions(competition_name)
    if sub.status == "pending" and (
        username_to_be_monitored is None or
        sub.submittedBy == username_to_be_monitored
    )
]

print(refs)
n_pending = len(refs)
while True:
    subs = [sub for sub in api.competition_submissions(competition_name) if sub.ref in refs]
    pending_subs = [sub for sub in subs if sub.status == "pending"]
    if len(pending_subs) != n_pending:
        print(f"{n_pending} -> {len(pending_subs)}")
        n_pending = len(pending_subs)
    if n_pending == 0:
        print("break")
        break
    for sub in subs:
        if sub.status == "pending":
            print(f"{sub.fileName} ({sub.ref}) by {sub.submittedBy}: {dt.datetime.utcnow() - sub.date}")

    sys.stdout.flush()
    time.sleep(60)
