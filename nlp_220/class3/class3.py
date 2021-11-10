import json

import pandas as pd

db_df = pd.read_csv("database.csv")
db_df.to_csv("database_without_header.csv", header=False, index=False)


db_1_df = db_df.sample(frac=0.5)
db_2_df = db_df.drop(db_1_df.index)

db_1_df.to_csv("database1.csv")
db_2_df.to_csv("database2.csv")


db_json = db_df.to_json(orient="records")
parsed_db_json = json.loads(db_json)
with open("database.json", "w") as db_json_file:
    json.dump(parsed_db_json, db_json_file, indent=4)


with open("EmployeeData.json") as employee_json_file:
    employee_df = pd.read_json(employee_json_file, orient="records")

employee_df.to_csv("EmployeeData.csv", index=False)
