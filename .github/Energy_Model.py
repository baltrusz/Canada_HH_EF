from Figaro_economic_data import figaro_economic_data

result = figaro_economic_data(
        figaro_folder=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data",
        first_year=2019,
        last_year=2019,
        edition_economic_data=2501,
    )
X = result["X"]
Y = result["Y"]
Z = result["Z"]

