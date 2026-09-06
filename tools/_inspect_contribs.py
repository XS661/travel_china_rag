import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("backend/data/contributions.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "select user_id, username, city, title, status, created_at "
    "from contributions where username='maptest_61920'"
).fetchall()
for row in rows:
    print(json.dumps({key: row[key] for key in row.keys()}, ensure_ascii=False))
