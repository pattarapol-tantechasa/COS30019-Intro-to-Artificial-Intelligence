import pandas as pd

# The SCATS xls has three sheets — 'Notes', 'Data', 'Summary Of Data'.
# We only care about 'Data'. Row 0 is a pre-header (just the time strings),
# row 1 is the actual column names, real data starts at row 2.
DATA_SHEET = "Data"


def load_raw(xls_path):
    """Read the Data sheet of the SCATS xls and return it as a DataFrame."""
    df = pd.read_excel(xls_path, sheet_name=DATA_SHEET, header=1)

    # SCATS Number comes in as a float — make it int so groupby keys are clean
    df['SCATS Number'] = df['SCATS Number'].astype('Int64')

    # The Date column has a dummy '00:15:00' time stamped onto each cell.
    # Strip that off so 'Date' is just a date.
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

    # Tidy whitespace
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype(str).str.strip()

    return df


def get_volume_cols(df):
    """Return the list of V00..V95 column names in order."""
    cols = [c for c in df.columns
            if isinstance(c, str) and c.startswith('V') and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def clean(df, vol_cols):
    """Drop unusable rows and patch small gaps in the volume columns."""
    df = df.dropna(subset=['SCATS Number', 'Date']).copy()
    df['SCATS Number'] = df['SCATS Number'].astype(int)

    # Negative volumes don't make sense — clip
    df[vol_cols] = df[vol_cols].clip(lower=0)

    # Fill small gaps within a day by interpolating across the row.
    # limit_direction='both' makes it also handle leading/trailing NaNs.
    df[vol_cols] = df[vol_cols].interpolate(axis=1, limit_direction='both')

    # If a whole day is still blank after interpolation, drop it
    before = len(df)
    df = df.dropna(subset=vol_cols, how='any')
    if before - len(df):
        print(f"  dropped {before - len(df)} rows that had unrecoverable gaps")

    return df.reset_index(drop=True)


def to_long(df, vol_cols):
    """
    Reshape from wide (one row per day, 96 volume columns) to long
    (one row per 15-min interval). Returns columns:
      SCATS Number, Location, NB_LATITUDE, NB_LONGITUDE, Timestamp, Volume
    """
    keep = ['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date']
    long = df.melt(id_vars=keep, value_vars=vol_cols,
                   var_name='Interval', value_name='Volume')

    # V00 -> 00:00, V01 -> 00:15, ..., V95 -> 23:45
    idx = long['Interval'].str[1:].astype(int)
    long['Timestamp'] = long['Date'] + pd.to_timedelta(idx * 15, unit='m')

    long = long.drop(columns=['Interval', 'Date'])
    long = long.sort_values(['SCATS Number', 'Timestamp']).reset_index(drop=True)
    return long
