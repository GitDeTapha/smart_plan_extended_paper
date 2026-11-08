# utils_prep.py
# Routines of the data preparation notebook (01_data_preparation.ipynb).
# Taken from utils_exploration_optimised.py of the conference paper; only the
# functions used by the notebook are kept. Every choice (paths, cutoff date,
# features) is passed as an argument from the "Paramètres" cell of the notebook.

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
from pandas.api.types import union_categoricals


############################################################
#                                                          #
#        ------  Data Understanding     -------            #
#                                                          #
############################################################

def load_raw_data(parquet_dir, valid_disp, stop_codes):
    """
    Load the raw consultations from the Parquet chunks written by
    scripts/csv_to_parquet.py (same columns as the original CSV), keeping only
    the consultations whose stop code is in `stop_codes`.

    The stop-code filter (section 3.1 of the paper's notebook) only looks at
    each row, so applying it while reading, chunk by chunk, gives the same rows
    as applying it after loading everything; but only about 15 M of the 52 M
    rows are then held in memory (all 52 M rows do not fit next to the other
    programs of a 16 GB laptop).

    Some lines of the CSV are malformed (a shifted column: disp = 'EU',
    AdministrativeFlag = 'ME', ...). The original notebook skipped them while
    reading the CSV (on_bad_lines='skip'); here they are the rows whose
    disposition is not one of `valid_disp`, and they are dropped.

    Args:
        parquet_dir (str): Folder containing the part_*.parquet files.
        valid_disp (list): The dispositions of a well-formed row.
        stop_codes (list): Stop codes kept (as integers).

    Returns:
        tuple: (DataFrame of the consultations kept, with activitydatetime as
        datetime; dict of row counts: read, malformed, well-formed, kept;
        Series of the dispositions of the well-formed rows).
    """
    # The Parquet files store the codes as text categories; they are turned
    # back into numbers as in the CSV, with small integer types.
    numeric_types = {"sta3n": "int16", "stopcode": "int16", "non_va": "int8", "zip": "float32"}
    keep = set(int(c) for c in stop_codes)
    parts, counts, disp_counts = [], {"read": 0, "malformed": 0, "well-formed": 0, "kept": 0}, []
    for path in sorted(glob.glob(f"{parquet_dir}/*.parquet")):
        part = pd.read_parquet(path)
        counts["read"] += len(part)
        well_formed = part["disp"].astype(str).isin(valid_disp)
        counts["malformed"] += int((~well_formed).sum())
        part = part[well_formed]
        counts["well-formed"] += len(part)
        disp_counts.append(part["disp"].astype(str).value_counts())
        part = part[pd.to_numeric(part["stopcode"].astype(str), errors="coerce").isin(keep)].copy()
        counts["kept"] += len(part)
        part["disp"] = part["disp"].astype(str).astype("category")
        # Converting chunk by chunk keeps memory low (a datetime takes 8 bytes,
        # the same date as a string about 60).
        part["activitydatetime"] = pd.to_datetime(part["activitydatetime"])
        for col, dtype in numeric_types.items():
            part[col] = pd.to_numeric(part[col].astype(str), errors="coerce").astype(dtype)
        parts.append(part)
    # Text columns stay categorical; their categories differ from one chunk to
    # the next, so they are merged first (otherwise pandas falls back to plain
    # strings, several GB on 52 M rows).
    text_cols = [c for c in parts[0].columns if isinstance(parts[0][c].dtype, pd.CategoricalDtype)]
    merged = {c: union_categoricals([p[c] for p in parts], ignore_order=True) for c in text_cols}
    data = pd.concat([p.drop(columns=text_cols) for p in parts], ignore_index=True)
    for c in text_cols:
        data[c] = merged[c]
    return data[parts[0].columns], counts, pd.concat(disp_counts).groupby(level=0).sum()


def extract_stop_codes(pdf_path):
    """
    Extract stop codes from a PDF document.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple: Lists of primary care codes, mental health codes, and all other specialties codes.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        primary_care_codes = []
        mental_health_codes = []
        all_other_specialties_codes = []

        # Read text from pages 2 to 4
        for page_num in range(1, 4):
            page = reader.pages[page_num]
            text = page.extract_text()

            # Extract stop codes based on section headers
            if 'Primary Care' in text:
                primary_care_section = text.split('Primary Care')[1].split('Mental Health')[0]
                primary_care_codes += [line.split()[0] for line in primary_care_section.split('\n') if line.strip()]

            if 'Mental Health' in text:
                mental_health_section = text.split('Mental Health')[1].split('All Other Specialties')[0]
                mental_health_codes += [line.split()[0] for line in mental_health_section.split('\n') if line.strip()]

            if 'All Other Specialties' in text:
                all_other_specialties_section = text.split('All Other Specialties')[1]
                all_other_specialties_codes += [line.split()[0] for line in all_other_specialties_section.split('\n') if line.strip()]

        return primary_care_codes, mental_health_codes, all_other_specialties_codes


def plot_histograms_numeric(data):
    """
    Plot histograms for the three delays dta, dts and dtc (in days).

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), dpi=72)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Histograms of Key Numeric Variables', fontsize=10)

    variables = ['dta', 'dts', 'dtc']
    xlims = [(0, 100), (0, 250), (0, 500)]

    for ax, var, xlim in zip(axes, variables, xlims):
        data[var].hist(bins=50, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {var}', fontsize=8)
        ax.set_xlabel(f'Values of {var} (days)', fontsize=8)
        ax.set_xlim(xlim)
        ax.tick_params(axis='both', which='major', labelsize=7)

    plt.tight_layout()
    plt.show()


def decompose_disp_column(data, disp_column="disp"):
    """
    Decompose 'disp' column into 'discontinued' and 'NoShow' columns.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        disp_column (str): Name of the 'disp' column to decompose.

    Returns:
        DataFrame: DataFrame with new columns added.
    """
    data['NoShow'] = data[disp_column].map({'DISCONTINUED': 0, 'CANCELLED': 1, 'COMPLETE/UPDATE': 0}).astype('uint8')
    data['discontinued'] = data[disp_column].map({'DISCONTINUED': 1, 'CANCELLED': 0, 'COMPLETE/UPDATE': 0}).astype('uint8')
    data.drop(disp_column, axis=1, inplace=True)
    return data


def plot_consultation_counts(data):
    """
    Plot the count of consultations for each category in 'NoShow'.

    Args:
        data (DataFrame): The pandas DataFrame containing the 'NoShow' column.
    """
    plt.figure(figsize=(4, 4), dpi=72)
    count_series = data['NoShow'].value_counts()
    plt.pie(count_series, labels=count_series.index, autopct='%1.2f%%', startangle=5, textprops={'fontsize': 8})
    plt.axis('equal')
    plt.title('Consultations per Category', fontsize=10)
    plt.tight_layout()
    plt.show()


def add_type_care(df, primary_care_codes):
    """Binary variable 'typecare': 0 = primary care, 1 = mental health or other specialty."""
    primary_care_codes = set(map(int, primary_care_codes))
    df['typecare'] = (~df['stopcode'].isin(primary_care_codes)).astype('uint8')
    return df


def filter_dates_after(data, datetime_col, cutoff_date):
    """
    Filter DataFrame to keep rows where datetime is after the cutoff date.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        datetime_col (str): Name of the datetime column.
        cutoff_date (str): The cutoff date in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        DataFrame: Filtered DataFrame.
    """
    return data[data[datetime_col] > pd.to_datetime(cutoff_date)]


def add_temporal_features(data, datetime_column):
    """
    Add temporal features from a datetime column.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        datetime_column (str): Name of the datetime column.

    Returns:
        DataFrame: DataFrame with new temporal features.
    """
    data['day_of_week'] = data[datetime_column].dt.dayofweek.astype('uint8')
    data['day_of_month'] = data[datetime_column].dt.day.astype('uint8')
    data['month'] = data[datetime_column].dt.month.astype('uint8')
    data['hour'] = data[datetime_column].dt.hour.astype('uint8')
    return data


def plot_patient_flow(data, disp_col='NoShow'):
    """
    Plot the no-show rate by hour, day of week and day of month.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        disp_col (str): Name of the column indicating show or no-show.
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=72)
    for ax, col, title in zip(axes, ['hour', 'day_of_week', 'day_of_month'],
                              ['Hour of Day', 'Day of the Week', 'Day of the Month']):
        rate = data.groupby(col)[disp_col].mean()
        ax.plot(rate.index, rate.values, marker='o', markersize=3)
        ax.set_title(f'No-Show Rate by {title}', fontsize=10)
        ax.set_xlabel(title, fontsize=8)
        ax.set_ylabel('Average No-Show Rate', fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 'day_of_week':
            ax.set_xticks(range(7))
            ax.set_xticklabels(days, rotation=45)
    plt.tight_layout()
    plt.show()


############################################################
#                                                          #
#        ------    Data Preparation     -------            #
#                                                          #
############################################################

def display_missing_values(data):
    """
    Display count and percentage of missing values per column.

    Args:
        data (DataFrame): The pandas DataFrame to analyze.

    Returns:
        Series: Percentages of missing values per column.
    """
    missing_values = data.isnull().sum()
    missing_values['state'] += (data['state'] == '*Missing*').sum()
    print("Percentage of missing values per column:")
    return (missing_values / data.shape[0]) * 100


def plot_no_show_rate_by(data, column, labels):
    """
    Plot the no-show rate (%) for the two values 0 and 1 of a binary column.

    Args:
        data (DataFrame): The consultations.
        column (str): Binary column, e.g. 'typecare' or 'non_va'.
        labels (list): Names of the values 0 and 1.
    """
    rates = data.groupby(column)['NoShow'].mean() * 100
    plt.figure(figsize=(5, 3.5), dpi=72)
    plt.bar(labels, rates.loc[[0, 1]].values, color=['#4682b4', '#add8e6'])
    plt.ylabel('No-Show Rate (%)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.show()


def add_prior_features(data, patient_col, datetime_col):
    """
    Add, for each consultation, the patient's history BEFORE this consultation:
    - prior_noShow_rate: share of no-shows among the previous consultations;
    - prior_service_time_avg: mean service_time of the previous consultations.
    A patient's first consultation gets 0 for both, as in the original notebook.

    Same values as the original `groupby().expanding()` followed by
    `shift(1)`, computed with cumulative sums (much faster on 10 M rows):
    sum of the previous values = cumulative sum - current value.

    Args:
        data (DataFrame): Consultations with NoShow and service_time.
        patient_col (str): Patient identifier column.
        datetime_col (str): Consultation date column.

    Returns:
        DataFrame: Sorted by patient then date, with the two new columns.
    """
    data = data.sort_values(by=[patient_col, datetime_col])
    n_previous = data.groupby(patient_col).cumcount()                     # 0 for the first consultation
    for target, new_col in [('NoShow', 'prior_noShow_rate'), ('service_time', 'prior_service_time_avg')]:
        # float64: NoShow (uint8) and the delays (uint16) would overflow when summed
        values = data[target].astype('float64')
        previous_sum = values.groupby(data[patient_col]).cumsum() - values
        data[new_col] = (previous_sum / n_previous.replace(0, np.nan)).fillna(0)
    return data
