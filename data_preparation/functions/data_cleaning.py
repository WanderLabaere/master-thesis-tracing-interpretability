"""
data_cleaning.py
----------------------------
Functions script to clean and format raw geochemical Excel datasets without altering underlying numerical values.

Input data:
1. Raw multi-element Excel datasets (ICPMS or XRF).

Generates and saves:
1. Fully cleaned and formatted complete datasets (CSV).
2. Isolated metadata feature datasets (CSV).
3. Isolated numerical feature datasets (CSV) with standardized decimal formatting and missing value representations.
"""

import pandas as pd
import os


def excel_to_clean_csv(excel_data_path, csv_out_path, csv_name, data_type):
    """
    In: excel file with OG data.
    Out: clean csv file, ready for handling.
    """

    # Read in the excel file
    df = pd.read_excel(excel_data_path)

    # Remove trailing/leading whitespaces in column names
    df.columns = df.columns.str.strip()

    # Remove trailing/leading whitespaces of strings in whole dataframe
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # rename '&' column name
    if data_type == "XRF":
        df = df.rename(columns={"&": "Ordernumber"})

    # Select metadata & numerical columns, based on data type.
    if data_type == "ICPMS":
        df_md = df.iloc[:, 0:16]
        df_num = df.iloc[:,16:]
    elif data_type == "XRF":
        df_md = df.iloc[:, 0:13]
        df_num = df.iloc[:,13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")
    
    # Clean up the numerical columns.
    for col in df_num.columns:
        df_num[col] = ( 
            df_num[col].astype(str)
                .str.replace(",", ".", regex=False)      # fix decimal comma
                .str.replace(" ", "", regex=False)       # remove spaces
                .replace({"": pd.NA, "nan": pd.NA, "-": pd.NA})
        )
        # Change data back to numerical
        # df_num[col] = pd.to_numeric(df_num[col], errors = "coerce") # coerce: non-numerical values are NA.

    # Concatenate back to complete datatframe
    df_full = pd.concat([df_md, df_num], axis = 1)

    # Create output paths & save metadata and numerical data independently
    full_output_path = os.path.join(csv_out_path, csv_name + ".csv") 
    md_output_path = os.path.join(csv_out_path, csv_name + "_md.csv")
    num_output_path = os.path.join(csv_out_path, csv_name + "_num.csv")

    df_full.to_csv(full_output_path, index = False, na_rep = "NA")
    df_md.to_csv(md_output_path, index = False, na_rep = "NA")
    df_num.to_csv(num_output_path, index = False, na_rep = "NA")
