import pathlib

import kaggle
import time
import datetime as dt
import sys


# competition_name = None
competition_name = "rsna-2022-cervical-spine-fracture-detection"
# username_to_be_monitored = "Ranchantan"
# username_to_be_monitored = None

api = kaggle.KaggleApi()
api.authenticate()

# refs = [
#     sub.ref
#     for sub in api.competition_submissions(competition_name)
#     if sub.status == "pending" and (
#         username_to_be_monitored is None or
#         sub.submittedBy == username_to_be_monitored
#     )
# ]
#
# print(refs)
# n_pending = len(refs)

old_refs = []
while True:
    pending_subs = [
        sub
        for sub in api.competition_submissions(competition_name)
        # if sub.ref in refs
        if sub.status == "pending"
    ]
    pending_refs = [sub.ref for sub in pending_subs]

    if len(pending_subs) != len(old_refs):
        print(f"{len(old_refs)} -> {len(pending_subs)}")
        if len(pending_subs) < len(old_refs):
            for sub in api.competition_submissions(competition_name):
                if sub.ref not in old_refs or sub.ref in pending_refs:
                    continue
                result = "\t".join([
                    str(sub.ref),
                    sub.fileName,
                    sub.description,
                    sub.submittedBy,
                    str(dt.datetime.utcnow() - sub.date).split('.')[0]
                ])
                print(f"* Completed: {result}")
                with open("results.tsv", "a") as f:
                    f.write(f"{result}\n")
                    f.flush()
        old_refs = pending_refs

    latest_status_fn = pathlib.Path("latest_status.txt")

    if len(pending_subs) > 0:
        msg_list = []
        for sub in pending_subs:
            if sub.status == "pending":
                msg = f"{sub.ref} {sub.fileName} by {sub.submittedBy} ({str(dt.datetime.utcnow() - sub.date).split('.')[0]})"
                print(msg)
                msg_list.append(msg)
        print(flush=True)

        with open(latest_status_fn, "w") as f:
            print("\n".join(msg_list), file=f)
    else:
        if latest_status_fn.exists():
            latest_status_fn.rename(latest_status_fn.parent / "_stored_last_status.txt")
    # sys.stdout.flush()
    time.sleep(60)
