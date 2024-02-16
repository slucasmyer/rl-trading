import csv
import os
import requests
from io import StringIO

API_KEY = "5W4QR93OQ48Z9GN6"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TQQQ&outputsize=full&datatype=csv&apikey={API_KEY}"

destination = "C:\\Users\\andre\\Desktop\\Class Folder\\Capstone"

response = requests.get(url)

if response.status_code == 200:
    # Use StringIO to create a file-like object from the CSV data
    csv_data = StringIO(response.text)

    # Use the csv module to read the CSV data
    reader = csv.reader(csv_data)

    row_list = []
    header = ["timestamp", "open", "high", "low", "close", "volume"]

    for row in reader:
        row_list.append(row)

    training_file_path = os.path.join(destination, "training_tqqq.csv")
    testing_file_path = os.path.join(destination, "testing_tqqq.csv")

    with open(training_file_path, mode="w", newline="") as training_file:
        training_writer = csv.writer(training_file)

        with open(testing_file_path, mode="w", newline="") as testing_file:
            testing_writer = csv.writer(testing_file)

            # Write the headers once in each file
            training_writer.writerow(header)
            testing_writer.writerow(header)

            for row in reversed(row_list):
                if row[0] == "timestamp":
                    break
                timestamp = row[0]
                year = int(timestamp[:4])
                if 2011 <= year <= 2021:
                    training_writer.writerow(row)
                elif 2022 <= year <= 2023:
                    testing_writer.writerow(row)

    print(f"CSV data downloaded and processed. Successfully created training_tqq.csv and testing_tqq.csv at "
          f"{destination}")
else:
    print(f"Failed to download CSV. Status code: {response.status_code}")
