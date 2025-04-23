"""
A short script to take an interjection result file and make a csv out of it
for upload to Google Sheets when validating the classificaitons.

Shuffles the results inside of each interjection.
"""

import argparse
import os
import csv

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename", required=True, type=str, help="The results file to output as a csv."
)
args = parser.parse_args()
results = pd.read_json(args.filename, lines=True)

results["interjection"] = results["interjection"].apply(str)

shuffled_df = (
    results.groupby("interjection", as_index=False)
    .apply(lambda x: x.sample(frac=1), include_groups=True)
    .sample(frac=1)
)

shuffled_df = shuffled_df.sort_values("interjection")

path = os.path.splitext(args.filename)[0]

out_file = f"{path}.csv"
shuffled_df.to_csv(out_file, index=False, quoting=csv.QUOTE_ALL)

print(f'Outputting to "{out_file}"')
