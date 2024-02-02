import csv
import requests
from io import StringIO

API_KEY = "5W4QR93OQ48Z9GN6"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TQQQ&datatype=csv&apikey={API_KEY}"
"""
Optional Parameters: 
'outputsize' - Default outputsize is 'compact'. Can do outputsize=full to receive stock's ENTIRE daily price history. 'compact' gives 100 most recent days
'datatype' - Can be set to either 'json' or 'csv'
"""


destination = "**Destination file path goes here**"

response = requests.get(url)

if response.status_code == 200:
    # Use StringIO to create a file-like object from the CSV data
    csv_data = StringIO(response.text)

    # Use the csv module to read the CSV data
    reader = csv.reader(csv_data)

    # # Process the CSV data, for example, print each row. Should comment this out if you're going to use outputsize=full
    for row in reader:
        print(row)

    # Reset the position of the StringIO object again before writing
    csv_data.seek(0)

    # Optionally, write the CSV data to a local file
    local_filename = f"{destination}\\output.csv"
    with open(local_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(reader)

    print(f"CSV data downloaded and processed. Local file created: {local_filename}")
else:
    print(f"Failed to download CSV. Status code: {response.status_code}")
