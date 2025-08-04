import pandas as pd
import urllib.request
import socket


# NOTES:
# We pick up information from all NYC 311 Service Requests, which is available from from 2010 to present. This information is automatically updated daily.
# MAIN URL with explanations, schema, etc: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data
# The API URL is: https://data.cityofnewyork.us/resource/erm2-nwe9.csv
# You do not need an API key for this public dataset.

year="2022"


# Check connectio to internet
def is_connected():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False
   


# replace spaces with "%20"
def download_311_data(year):
    url = (
        "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?"
        f"$where=created_date%20between%20'{year}-01-01T00:00:00'%20and%20'{year}-12-31T23:59:59'"
    )
    output_filename = f"311_Service_Requests_{year}.csv"


    if not is_connected():
        print("No internet connection. Please check your network.")
        return


    try:
        print("Downloading 311 data from NYC Open Data portal for year: {year}...")
        df = pd.read_csv(url, parse_dates=["created_date", "closed_date"], low_memory=False)
        print(f"Downloaded {len(df)} records.")


        df.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
    except Exception as e:
        print(f"Error downloading or processing data: {e}")


if __name__ == "__main__":
    download_311_data(year)



