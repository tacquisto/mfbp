# segy_io.py
"""
I/O functions for reading seismic data, source wavelets, and traveltime picks.
"""

import numpy as np
import pandas as pd
import obspy


def read_segy(filepath):
    """
    Read a SEG-Y file and return an ObsPy Stream.

    Parameters
    ----------
    filepath : str
        Path to the SEG-Y file.

    Returns
    -------
    stream : obspy.Stream
        Stream containing all traces.
    """
    stream = obspy.read(filepath, format="SEGY")
    return stream


def read_wavelet(filepath, fmt="csv"):
    """
    Read a source wavelet from file.

    Parameters
    ----------
    filepath : str
        Path to the wavelet file.
    fmt : str
        File format: 'csv' or 'binary'.
        Both formats expect two columns: time (s) and amplitude.

    Returns
    -------
    time : np.ndarray
        Time vector in seconds.
    amplitude : np.ndarray
        Wavelet amplitude.
    """
    if fmt == "csv":
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    elif fmt == "binary":
        data = np.fromfile(filepath, dtype=np.float64).reshape(-1, 2)
    else:
        raise ValueError(f"Unsupported wavelet format: {fmt}. Use 'csv' or 'binary'.")

    time = data[:, 0]
    amplitude = data[:, 1]
    return time, amplitude


def read_picks(filepath):
    """
    Read traveltime picks from a CSV file.

    Required columns: TIME, FFID, CHANNEL
        - TIME: first-arrival time in seconds
        - FFID: field file ID (shot number)
        - CHANNEL: channel/trace number

    Parameters
    ----------
    filepath : str
        Path to the picks CSV file.

    Returns
    -------
    picks : pd.DataFrame
        DataFrame with columns TIME, FFID, CHANNEL.
    """
    picks = pd.read_csv(filepath)

    required = {"TIME", "FFID", "CHANNEL"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"Picks file missing required columns: {missing}")

    return picks


def get_gather(stream, ffid):
    """
    Extract a single shot gather from a stream by FFID.

    Parameters
    ----------
    stream : obspy.Stream
        Full stream containing multiple shots.
    ffid : int
        Field file ID to extract.

    Returns
    -------
    gather : obspy.Stream
        Stream containing only traces matching the FFID.
    """
    gather = stream.select(id=None)  # empty stream
    for tr in stream:
        if tr.stats.segy.trace_header.FieldRecord == ffid:
            gather.append(tr)
    return gather


def get_picks_for_gather(picks, ffid):
    """
    Get picks for a specific shot gather.

    Parameters
    ----------
    picks : pd.DataFrame
        Full picks DataFrame.
    ffid : int
        Field file ID to filter by.

    Returns
    -------
    gather_picks : pd.DataFrame
        Picks for the specified FFID, sorted by CHANNEL.
    """
    gather_picks = picks[picks["FFID"] == ffid].sort_values("CHANNEL")
    return gather_picks