"""
Event Study Calculation Engine Module.

This module provides the EventStudy class, which is the core computational engine
for calculating Cumulative Abnormal Returns (CAR) and Buy-and-Hold Abnormal
Returns (BHAR) using various risk models. This is the statistical workhorse of
the event study framework.

Architecture Overview
---------------------
The downstream event study functionality is split across two modules:

1. **event_study.py**: ThematicES class
   - Orchestrates the full event study workflow
   - Pulls fundamental data from WRDS (Compustat, CRSP, IBES)
   - Computes financial covariates for regression
   - Delegates CAR/BHAR computation to this module

2. **This module (event_study_module.py)**: EventStudy class
   - Core computational engine for abnormal returns
   - Implements multiple risk models
   - Handles estimation windows and event windows
   - Pure statistical computation, no data orchestration

Why Two Separate Files?
-----------------------
These files are intentionally separate because they serve different purposes:

- **Separation of Concerns**: ThematicES handles data orchestration and business
  logic, while EventStudy handles pure statistical computation.

- **Reusability**: EventStudy can be used independently for any event study,
  not just thematic sentiment analysis.

- **Testability**: EventStudy can be tested in isolation with mock data,
  without requiring WRDS covariate queries.

Risk Models Supported
---------------------
The EventStudy class supports four risk adjustment models:

- **'madj' (Market-Adjusted)**: Excess returns over market return.
  Simple and robust, minimal estimation error. Default for thematic studies.

- **'m' (Market Model)**: Single-factor CAPM model.
  ret = alpha + beta * mktrf + epsilon

- **'ff' (Fama-French Three-Factor)**: Industry-standard academic model.
  ret = alpha + beta1 * mktrf + beta2 * smb + beta3 * hml + epsilon

- **'ffm' (Fama-French Four-Factor with Momentum)**: Extended model.
  ret = alpha + beta1 * mktrf + beta2 * smb + beta3 * hml + beta4 * umd + epsilon

Typical Usage
-------------
For direct CAR/BHAR calculation::

    from src.event_study_module import EventStudy

    events = [
        {"permno": 10002, "edate": "05/29/2012"},
        {"permno": 14593, "edate": "06/15/2012"},
    ]

    # Initialize calculator
    es = EventStudy(output_path="/path/to/output")

    # Run event study with market-adjusted model
    results = es.eventstudy(
        data=events,
        model='madj',       # Risk model
        estwin=100,         # Estimation window (days)
        gap=50,             # Gap before event window
        evtwins=-10,        # Event window start
        evtwine=10,         # Event window end
        minval=70,          # Minimum observations required
        output='df'         # Return format
    )

    # Results contains three DataFrames:
    # - event_stats: Cross-sectional statistics across events
    # - event_window: Daily returns within event window
    # - event_date: CAR/BHAR at event date for each firm

Output Formats
--------------
The eventstudy() method supports multiple output formats:

- **'df'**: Dictionary of pandas DataFrames (default)
- **'print'**: Console output
- **'json'**: JSON file and string
- **'csv'**: CSV file and string
- **'xls'**: Excel file

Key Output Columns
------------------
The event_date DataFrame includes:

- **cret**: Cumulative raw return over event window
- **car**: Cumulative Abnormal Return
- **bhar**: Buy-and-Hold Abnormal Return
- **scar**: Standardized CAR
- **sar**: Standardized Abnormal Return

WRDS Data Requirements
----------------------
The EventStudy class queries WRDS for:

- **CRSP Daily Stock File (crsp_a_stock.dsf)**: Daily returns
- **CRSP Daily Index (crsp_a_stock.dsi)**: Trading calendar
- **Fama-French Factors (ff_all.factors_daily)**: Risk factors
- **CRSP Delisting Returns (crsp_a_stock.dsedelist)**: Delisting adjustments

See Also
--------
- event_study.ThematicES: Thematic event study orchestrator
- wrds_connection.WRDSConnection: Context manager for WRDS connections
"""
# Import the required packages
from datetime import datetime, date
from io import StringIO as StringIO_StringIO
from json import (
    dumps as json_dumps,
    dump as json_dump,
    load as json_load,
    JSONEncoder as json_JSONEncoder,
)

import json
import os

from pandas import (
    DataFrame as pd_DataFrame,
    ExcelWriter as pd_ExcelWriter,
)
from numpy import (
    abs as np_abs,
    nan as np_nan,
    mean as np_mean,
    std as np_std,
    sqrt as np_sqrt,
    ndarray as np_ndarray,
)

from statsmodels.api import (
    OLS as sm_OLS,
    add_constant as sm_add_constant
)
import logging

from src.wrds_connection import WRDSConnection


class EncoderJson(json_JSONEncoder):
    """
    Class used to encodes to JSON data format.
    """

    def default(self, obj):
        if isinstance(obj, np_ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        elif isinstance(obj, date):
            return obj.__str__()

        return json_JSONEncoder.default(self, obj)


class EventStudy(object):
    """
    Core CAR/BHAR calculation engine for event studies.

    This class is the computational workhorse that calculates cumulative abnormal
    returns (CAR) and buy-and-hold abnormal returns (BHAR) using various risk models:
    - Market-adjusted model (madj)
    - Market model (m)
    - Fama-French three-factor model (ff)
    - Fama-French four-factor model with momentum (ffm)

    Note: This class handles the statistical calculations. For thematic event study
    orchestration (pulling WRDS covariates, preparing regression data), see
    ThematicES in event_study.py.

    Args:
        output_path: Path for output files (logs, results). Defaults to home directory.
        wrds_connection: Optional existing WRDS connection. If provided, the
            connection will be reused and NOT closed by this class.
    """

    def __init__(self, output_path='', wrds_connection=None):
        """
        Initialize the EventStudy calculator.

        Args:
            output_path: Path for output files. Defaults to home directory.
            wrds_connection: Optional existing WRDS connection to reuse.
        """
        if len(output_path) <= 0:
            self.output_path = os.path.expanduser('~')
        else:
            self.output_path = output_path

        # Store the provided connection for use with WRDSConnection context manager
        self.wrds_connection = wrds_connection

        # Add a flag to indicate whether a message about data issues needs to be printed
        self.has_data_issues = False

        # Add logging to capture potential data issues
        lfpath = os.path.join(self.output_path, "EventStudy.log")
        logging.basicConfig(filename=lfpath, filemode='w', level=logging.DEBUG)

    def eventstudy(self, data=None, model='m', estwin=100, gap=50, evtwins=-10, evtwine=10, minval=70, output='df'):
        """
        Run the event study calculation.

        Args:
            data: Event data (list of dicts with 'edate' and 'permno' keys).
            model: Risk model to use:
                - 'madj': Market-adjusted model
                - 'm': Market model
                - 'ff': Fama-French three-factor
                - 'ffm': Fama-French four-factor with momentum
            estwin: Estimation window length (default: 100 days).
            gap: Gap between estimation window and event window (default: 50 days).
            evtwins: Days before event date to begin event window (default: -10).
            evtwine: Days after event date to end event window (default: 10).
            minval: Minimum non-missing return observations required (default: 70).
            output: Output format:
                - 'df': Dictionary of pandas DataFrames
                - 'xls': Excel file
                - 'csv': CSV file
                - 'json': JSON file
                - 'print': Console output

        Returns:
            Results in the specified format.
        """
        # Calculate window parameters
        estwins = (estwin + gap + np_abs(evtwins))
        estwine = (gap + np_abs(evtwins) + 1)
        evtwinx = (estwins + 1)
        evtwins = np_abs(evtwins)
        evtrang = (evtwins + evtwine + 1)

        # Default event data if not provided
        evtdata = [{"edate": "05/29/2012", "permno": "10002"}]
        if data is not None:
            evtdata = json_dumps(data)

        params = {
            'estwins': estwins,
            'estwine': estwine,
            'evtwins': evtwins,
            'evtwine': evtwine,
            'evtwinx': evtwinx,
            'evtdata': evtdata
        }

        # Use context manager for WRDS connection
        with WRDSConnection(connection=self.wrds_connection) as wconn:
            # Get the initial data from the database
            df = wconn.raw_sql("""
            SELECT
                    a.*,
                    x.*,
                    c.date as rdate,
                    c.ret as ret1,
                    (f.mktrf+f.rf) as mkt,
                    f.mktrf,
                    f.rf,
                    f.smb,
                    f.hml,
                    f.umd,
                    (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1-(f.mktrf+f.rf) as exret,
                    (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1 as ret,
                    case when c.date between a.estwin1 and a.estwin2 then 1 else 0 end as isest,
                    case when c.date between a.evtwin1 and a.evtwin2 then 1 else 0 end as isevt,
                    case
                      when c.date between a.evtwin1 and a.evtwin2 then (rank() OVER (PARTITION BY x.evtid ORDER BY c.date)-%(evtwinx)s)
                      else (rank() OVER (PARTITION BY x.evtid ORDER BY c.date))
                    end as evttime,
                    case
                      when c.date = a.date then 1
                      else 0
                    end as evtflag
            FROM
              (
                SELECT
                  date,
                  lag(date, %(estwins)s ) over (order by date) as estwin1,
                  lag(date, %(estwine)s )  over (order by date) as estwin2,
                  lag(date, %(evtwins)s )  over (order by date) as evtwin1,
                  lead(date, %(evtwine)s )  over (order by date) as evtwin2
                FROM crsp_a_stock.dsi
              ) as a
            JOIN
            (select
                    to_char(x.edate, 'ddMONYYYY') || trim(to_char(x.permno,'999999999')) as evtid,
                    x.permno,
                    x.edate
            from
            json_to_recordset('%(evtdata)s') as x(edate date, permno int)
            ) as x
              ON a.date=x.edate
            JOIN crsp_a_stock.dsf c
                ON x.permno=c.permno
                AND c.date BETWEEN a.estwin1 and a.evtwin2
            JOIN ff_all.factors_daily f
                ON c.date=f.date
            LEFT JOIN crsp_a_stock.dsedelist d
                ON x.permno=d.permno
                AND c.date=d.dlstdt
            WHERE f.mktrf is not null
            AND c.ret is not null
            ORDER BY x.evtid, x.permno, a.date, c.date
            """ % params)

        # Columns coming from the database query
        df.columns = ['date', 'estwin1', 'estwin2', 'evtwin1', 'evtwin2',
                      'evtid', 'permno', 'edate', 'rdate', 'ret1', 'mkt',
                      'mktrf', 'rf', 'smb', 'hml', 'umd', 'exret', 'ret',
                      'isest', 'isevt', 'evttime', 'evtflag']

        # Convert nullable Float64 dtypes to regular float64 for statsmodels compatibility
        numeric_cols = ['ret1', 'mkt', 'mktrf', 'rf', 'smb', 'hml', 'umd', 'exret', 'ret']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'Float64':
                df[col] = df[col].astype('float64')

        # Additional columns that will hold computed values
        addcols = ['RMSE', 'INTERCEPT', 'var_estp', 'expret', 'abret',
                   'alpha', '_nobs', '_p_', '_edf_', 'rsq', 'cret',
                   'cexpret', 'car', 'scar', 'sar', 'pat_scale', 'bhar',
                   'lastevtwin', 'cret_edate', 'scar_edate', 'car_edate',
                   'bhar_edate', 'pat_scale_edate', 'xyz']

        for c in addcols:
            if c == 'lastevtwin':
                df[c] = 0
            else:
                df[c] = np_nan

        # Process each event
        for evt in data:
            permno = evt['permno']
            xdate = evt['edate']
            edate = datetime.strptime(xdate, "%m/%d/%Y").date().strftime("%Y-%m-%d")

            ths_mask = (df['permno'] == permno) & (df['edate'] == edate)
            est_mask = (df['permno'] == permno) & (df['edate'] == edate) & (df['isest'] == 1)
            evt_mask = (df['permno'] == permno) & (df['edate'] == edate) & (df['isevt'] == 1)
            flg_mask = (df['permno'] == permno) & (df['edate'] == edate) & (df['evtflag'] == 1)
            err_mask1 = (df['permno'] == permno) & (df['edate'] == edate) & (df['isevt'] == 1) & (df['evttime'] < -1*evtwins)
            err_mask2 = (df['permno'] == permno) & (df['edate'] == edate) & (df['isevt'] == 1) & (df['evttime'] > evtwine)

            # Check data requirements
            _nobs = df["ret"][est_mask].count()
            _flgs = df["ret"][flg_mask].count()
            _wins = df["ret"][evt_mask].count()

            _needs_fix = False
            if df["evttime"][err_mask1].count() > 0 or df["evttime"][err_mask2].count() > 0:
                _needs_fix = True

            # Fix evttime values if needed
            if _needs_fix:
                evtrow = df[flg_mask]
                evtcnt = df[flg_mask]["evttime"].count()
                if evtcnt > 0:
                    evtidx = list(evtrow.index)[0]

                    def fixEvtWinTime(row):
                        return int(row.name.__int__())-evtidx

                    df.loc[evt_mask, "evttime"] = df.loc[evt_mask].apply(fixEvtWinTime, axis=1)

            # Process event if data requirements are met
            if (_nobs >= minval) and (_flgs > 0) and (_wins == evtrang):

                # Market-Adjusted Model
                if model == 'madj':
                    y = df["exret"][est_mask]
                    mean = np_mean(y)
                    stdv = np_std(y, ddof=1)

                    df.loc[evt_mask, 'INTERCEPT'] = mean
                    df.loc[evt_mask, 'RMSE'] = stdv
                    df.loc[evt_mask, '_nobs'] = len(y)
                    df.loc[evt_mask, 'var_estp'] = stdv ** 2
                    df.loc[evt_mask, 'alpha'] = mean
                    df.loc[evt_mask, 'rsq'] = 0
                    df.loc[evt_mask, '_p_'] = 1
                    df.loc[evt_mask, '_edf_'] = (len(y) - 1)
                    df.loc[evt_mask, 'expret'] = df.loc[evt_mask, 'mkt']
                    df.loc[evt_mask, 'abret'] = df.loc[evt_mask, 'exret']
                    df_est = df[est_mask]
                    _nobs = len(df_est[df_est.ret.notnull()])

                    nloc = {'const': 0}

                    def f_cret(row):
                        tmp = ((row['ret'] * nloc['const']) + (row['ret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cret'] = df[evt_mask].apply(f_cret, axis=1)
                    df.loc[evt_mask, 'cret_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_cexpret(row):
                        tmp = ((row['expret'] * nloc['const']) + (row['expret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cexpret'] = df[evt_mask].apply(f_cexpret, axis=1)

                    nloc = {'const': 0}

                    def f_car(row):
                        tmp = (row['abret'] + nloc['const'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'car'] = df[evt_mask].apply(f_car, axis=1)
                    df.loc[evt_mask, 'car_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_sar(row):
                        tmp = (row['abret'] / np_sqrt(row['var_estp']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'sar'] = df[evt_mask].apply(f_sar, axis=1)
                    df.loc[evt_mask, 'sar_edate'] = nloc['const']

                    nloc = {'const': 0, 'evtrang': evtrang}

                    def f_scar(row):
                        tmp = (row['car'] / np_sqrt((evtrang * row['var_estp'])))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'scar'] = df[evt_mask].apply(f_scar, axis=1)
                    df.loc[evt_mask, 'scar_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_bhar(row):
                        tmp = (row['cret'] - row['cexpret'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'bhar'] = df[evt_mask].apply(f_bhar, axis=1)
                    df.loc[evt_mask, 'bhar_edate'] = nloc['const']

                    df.loc[evt_mask, 'pat_scale'] = (_nobs - 2.00) / (_nobs - 4.00)
                    df.loc[evt_mask, 'pat_scale_edate'] = (_nobs - 2.00) / (_nobs - 4.00)

                # Market Model
                elif model == 'm':
                    X = df["mktrf"][est_mask]
                    y = df["ret"][est_mask]

                    X = sm_add_constant(X)
                    est = sm_OLS(y, X).fit()

                    df_est = df[(df['permno'] == permno) & (df['edate'] == edate) & (df['isest'] == 1)]
                    _nobs = len(df_est[df_est.ret.notnull()])

                    alpha = est.params.__getitem__('const')
                    beta1 = est.params.__getitem__('mktrf')

                    df.loc[evt_mask, 'INTERCEPT'] = alpha
                    df.loc[evt_mask, 'alpha'] = alpha
                    df.loc[evt_mask, 'RMSE'] = np_sqrt(est.mse_resid)
                    df.loc[evt_mask, '_nobs'] = _nobs
                    df.loc[evt_mask, 'var_estp'] = est.mse_resid
                    df.loc[evt_mask, 'rsq'] = est.rsquared
                    df.loc[evt_mask, '_p_'] = 2
                    df.loc[evt_mask, '_edf_'] = (len(y) - 2)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'const': 0}

                    def f_expret(row):
                        return (nloc['alpha'] + (nloc['beta1'] * row['mktrf']))
                    df.loc[evt_mask, 'expret'] = df[evt_mask].apply(f_expret, axis=1)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'const': 0}

                    def f_abret(row):
                        return (row['ret'] - (nloc['alpha'] + (nloc['beta1'] * row['mktrf'])))
                    df.loc[evt_mask, 'abret'] = df[evt_mask].apply(f_abret, axis=1)

                    nloc = {'const': 0}

                    def f_cret(row):
                        tmp = ((row['ret'] * nloc['const']) + (row['ret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cret'] = df[evt_mask].apply(f_cret, axis=1)
                    df.loc[evt_mask, 'cret_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_cexpret(row):
                        tmp = ((row['expret'] * nloc['const']) + (row['expret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cexpret'] = df[evt_mask].apply(f_cexpret, axis=1)

                    nloc = {'const': 0}

                    def f_car(row):
                        tmp = (row['abret'] + nloc['const'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'car'] = df[evt_mask].apply(f_car, axis=1)
                    df.loc[evt_mask, 'car_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_sar(row):
                        tmp = (row['abret'] / np_sqrt(row['var_estp']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'sar'] = df[evt_mask].apply(f_sar, axis=1)
                    df.loc[evt_mask, 'sar_edate'] = nloc['const']

                    nloc = {'const': 0, 'evtrang': evtrang}

                    def f_scar(row):
                        tmp = (row['car'] / np_sqrt((evtrang * row['var_estp'])))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'scar'] = df[evt_mask].apply(f_scar, axis=1)
                    df.loc[evt_mask, 'scar_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_bhar(row):
                        tmp = (row['cret'] - row['cexpret'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'bhar'] = df[evt_mask].apply(f_bhar, axis=1)
                    df.loc[evt_mask, 'bhar_edate'] = nloc['const']

                    df.loc[evt_mask, 'pat_scale'] = (_nobs - 2.00) / (_nobs - 4.00)
                    df.loc[evt_mask, 'pat_scale_edate'] = (_nobs - 2.00) / (_nobs - 4.00)

                # Fama-French Three Factor Model
                elif model == 'ff':
                    df_est = df[(df['permno'] == permno) & (df['edate'] == edate) & (df['isest'] == 1)]
                    X = df_est[['smb', 'hml', 'mktrf']]
                    y = df_est['ret']

                    X = sm_add_constant(X)
                    est = sm_OLS(y, X).fit()

                    alpha = est.params.__getitem__('const')
                    beta1 = est.params.__getitem__('mktrf')
                    beta2 = est.params.__getitem__('smb')
                    beta3 = est.params.__getitem__('hml')

                    df.loc[evt_mask, 'INTERCEPT'] = alpha
                    df.loc[evt_mask, 'alpha'] = alpha
                    df.loc[evt_mask, 'RMSE'] = np_sqrt(est.mse_resid)
                    df.loc[evt_mask, '_nobs'] = _nobs
                    df.loc[evt_mask, 'var_estp'] = est.mse_resid
                    df.loc[evt_mask, 'rsq'] = est.rsquared
                    df.loc[evt_mask, '_p_'] = 2
                    df.loc[evt_mask, '_edf_'] = (len(y) - 2)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3, 'const': 0}

                    def f_expret(row):
                        return ((nloc['alpha'] + (nloc['beta1'] * row['mktrf']) + (nloc['beta2'] * row['smb']) + (nloc['beta3'] * row['hml'])))
                    df.loc[evt_mask, 'expret'] = df[evt_mask].apply(f_expret, axis=1)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3, 'const': 0}

                    def f_abret(row):
                        return (row['ret'] - ((nloc['alpha'] + (nloc['beta1'] * row['mktrf']) + (nloc['beta2'] * row['smb']) + (nloc['beta3'] * row['hml']))))
                    df.loc[evt_mask, 'abret'] = df[evt_mask].apply(f_abret, axis=1)

                    nloc = {'const': 0}

                    def f_cret(row):
                        tmp = ((row['ret'] * nloc['const']) + (row['ret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cret'] = df[evt_mask].apply(f_cret, axis=1)
                    df.loc[evt_mask, 'cret_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_cexpret(row):
                        tmp = ((row['expret'] * nloc['const']) + (row['expret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cexpret'] = df[evt_mask].apply(f_cexpret, axis=1)
                    nloc = {'const': 0}

                    def f_car(row):
                        tmp = (row['abret'] + nloc['const'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'car'] = df[evt_mask].apply(f_car, axis=1)
                    df.loc[evt_mask, 'car_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_sar(row):
                        tmp = (row['abret'] / np_sqrt(row['var_estp']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'sar'] = df[evt_mask].apply(f_sar, axis=1)
                    df.loc[evt_mask, 'sar_edate'] = nloc['const']

                    nloc = {'const': 0, 'evtrang': evtrang}

                    def f_scar(row):
                        tmp = (row['car'] / np_sqrt((evtrang * row['var_estp'])))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'scar'] = df[evt_mask].apply(f_scar, axis=1)
                    df.loc[evt_mask, 'scar_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_bhar(row):
                        tmp = (row['cret'] - row['cexpret'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'bhar'] = df[evt_mask].apply(f_bhar, axis=1)
                    df.loc[evt_mask, 'bhar_edate'] = nloc['const']

                    df.loc[evt_mask, 'pat_scale'] = (_nobs - 2.00) / (_nobs - 4.00)
                    df.loc[evt_mask, 'pat_scale_edate'] = (_nobs - 2.00) / (_nobs - 4.00)

                # Fama-French Plus Momentum
                elif model == 'ffm':
                    df_est = df[(df['permno'] == permno) & (df['edate'] == edate) & (df['isest'] == 1)]

                    X = df_est[['mktrf', 'smb', 'hml', 'umd']]
                    y = df_est['ret']

                    X = sm_add_constant(X)
                    est = sm_OLS(y, X).fit()

                    alpha = est.params.__getitem__('const')
                    beta1 = est.params.__getitem__('mktrf')
                    beta2 = est.params.__getitem__('smb')
                    beta3 = est.params.__getitem__('hml')
                    beta4 = est.params.__getitem__('umd')

                    df.loc[evt_mask, 'INTERCEPT'] = alpha
                    df.loc[evt_mask, 'alpha'] = alpha
                    df.loc[evt_mask, 'RMSE'] = np_sqrt(est.mse_resid)
                    df.loc[evt_mask, '_nobs'] = _nobs
                    df.loc[evt_mask, 'var_estp'] = est.mse_resid
                    df.loc[evt_mask, 'rsq'] = est.rsquared
                    df.loc[evt_mask, '_p_'] = 2
                    df.loc[evt_mask, '_edf_'] = (len(y) - 2)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3, 'beta4': beta4, 'const': 0}

                    def f_expret(row):
                        return ((nloc['alpha'] + (nloc['beta1'] * row['mktrf']) + (nloc['beta2'] * row['smb']) + (nloc['beta3'] * row['hml']) + (nloc['beta4'] * row['umd'])))
                    df.loc[evt_mask, 'expret'] = df[evt_mask].apply(f_expret, axis=1)

                    nloc = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3, 'beta4': beta4, 'const': 0}

                    def f_abret(row):
                        return (row['ret'] - ((nloc['alpha'] + (nloc['beta1'] * row['mktrf']) + (nloc['beta2'] * row['smb']) + (nloc['beta3'] * row['hml']) + (nloc['beta4'] * row['umd']))))
                    df.loc[evt_mask, 'abret'] = df[evt_mask].apply(f_abret, axis=1)

                    nloc = {'const': 0}

                    def f_cret(row):
                        tmp = ((row['ret'] * nloc['const']) + (row['ret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cret'] = df[evt_mask].apply(f_cret, axis=1)
                    df.loc[evt_mask, 'cret_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_cexpret(row):
                        tmp = ((row['expret'] * nloc['const']) + (row['expret'] + nloc['const']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'cexpret'] = df[evt_mask].apply(f_cexpret, axis=1)
                    nloc = {'const': 0}

                    def f_car(row):
                        tmp = (row['abret'] + nloc['const'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'car'] = df[evt_mask].apply(f_car, axis=1)
                    df.loc[evt_mask, 'car_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_sar(row):
                        tmp = (row['abret'] / np_sqrt(row['var_estp']))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'sar'] = df[evt_mask].apply(f_sar, axis=1)
                    df.loc[evt_mask, 'sar_edate'] = nloc['const']

                    nloc = {'const': 0, 'evtrang': evtrang}

                    def f_scar(row):
                        tmp = (row['car'] / np_sqrt((evtrang * row['var_estp'])))
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'scar'] = df[evt_mask].apply(f_scar, axis=1)
                    df.loc[evt_mask, 'scar_edate'] = nloc['const']

                    nloc = {'const': 0}

                    def f_bhar(row):
                        tmp = (row['cret'] - row['cexpret'])
                        nloc['const'] = tmp
                        return tmp
                    df.loc[evt_mask, 'bhar'] = df[evt_mask].apply(f_bhar, axis=1)
                    df.loc[evt_mask, 'bhar_edate'] = nloc['const']

                    df.loc[evt_mask, 'pat_scale'] = (_nobs - 2.00) / (_nobs - 4.00)
                    df.loc[evt_mask, 'pat_scale_edate'] = (_nobs - 2.00) / (_nobs - 4.00)

                else:
                    df['isest'][evt_mask] = -2

            else:
                # Event failed data checks
                self.has_data_issues = True
                df.drop(df.loc[ths_mask].index, inplace=True)
                logging.warning("\nFailed data check: " + str(evt) + "\n"
                                "Estimation Window Obs: " + str(_nobs) + "; Min Required: " + str(minval) + "\n" +
                                "Event Date Flag: " + str(_flgs) + "\n" +
                                "Event Window Obs: " + str(_wins) + "; Expected: " + str(evtrang) + "\n" +
                                "--------------------" + "\n")

        # Warn about data issues
        if self.has_data_issues:
            print("NOTE: Some data related issues were encountered. Please consult the log file ("+os.path.join(self.output_path, 'EventStudy.log')+").")

        # Prepare output statistics
        df_sta = df[df['isevt'] == 1]
        levt = df_sta['evttime'].unique()

        columns = ['evttime',
                   'car_m',
                   'ret_m',
                   'abret_m',
                   'abret_t',
                   'sar_t',
                   'pat_ar',
                   'cret_edate_m',
                   'car_edate_m',
                   'pat_car_edate_m',
                   'car_edate_t',
                   'scar_edate_t',
                   'bhar_edate_m']

        idxlist = list(levt)
        df_stats = pd_DataFrame(index=idxlist, columns=columns)
        df_stats = df_stats.fillna(0.00000000)

        # Event
        df_stats['evttime'] = df_sta.groupby(['evttime'])['evttime'].unique()
        # Means
        df_stats['abret_m'] = df_sta.groupby(['evttime'])['abret'].mean()
        df_stats['bhar_edate_m'] = df_sta.groupby(['evttime'])['bhar_edate'].mean()
        df_stats['car_edate_m'] = df_sta.groupby(['evttime'])['car_edate'].mean()
        df_stats['car_m'] = df_sta.groupby(['evttime'])['car'].mean()
        df_stats['cret_edate_m'] = df_sta.groupby(['evttime'])['cret_edate'].mean()
        df_stats['pat_scale_m'] = df_sta.groupby(['evttime'])['pat_scale'].mean()
        df_stats['pat_car_edate_mean'] = 0
        df_stats['ret_m'] = df_sta.groupby(['evttime'])['ret'].mean()
        df_stats['sar_m'] = df_sta.groupby(['evttime'])['sar'].mean()
        df_stats['scar_edate_m'] = df_sta.groupby(['evttime'])['scar_edate'].mean()
        df_stats['scar_m'] = df_sta.groupby(['evttime'])['scar'].mean()
        # Standard deviations
        df_stats['car_v'] = df_sta.groupby(['evttime'])['car'].std()
        df_stats['abret_v'] = df_sta.groupby(['evttime'])['abret'].std()
        df_stats['sar_v'] = df_sta.groupby(['evttime'])['sar'].std()
        df_stats['pat_scale_v'] = df_sta.groupby(['evttime'])['pat_scale'].std()
        df_stats['car_edate_v'] = df_sta.groupby(['evttime'])['car_edate'].std()
        df_stats['scar_edate_v'] = df_sta.groupby(['evttime'])['scar_edate'].std()
        df_stats['scar_v'] = df_sta.groupby(['evttime'])['scar'].std()
        # Counts
        df_stats['scar_n'] = df_sta.groupby(['evttime'])['scar'].count()
        df_stats['scar_edate_n'] = df_sta.groupby(['evttime'])['scar_edate'].count()
        df_stats['sar_n'] = df_sta.groupby(['evttime'])['sar'].count()
        df_stats['car_n'] = df_sta.groupby(['evttime'])['car'].count()
        df_stats['n'] = df_sta.groupby(['evttime'])['evttime'].count()
        # Sums
        df_stats['pat_scale_edate_s'] = df_sta.groupby(['evttime'])['pat_scale_edate'].sum()
        df_stats['pat_scale_s'] = df_sta.groupby(['evttime'])['pat_scale'].sum()

        # T statistics
        def tstat(row, m, v, n):
            return row[m] / (row[v] / np_sqrt(row[n]))

        df_stats['abret_t'] = df_stats.apply(tstat, axis=1, args=('abret_m', 'abret_v', 'n'))
        df_stats['sar_t'] = df_stats.apply(tstat, axis=1, args=('sar_m', 'sar_v', 'n'))
        df_stats['car_edate_t'] = df_stats.apply(tstat, axis=1, args=('car_edate_m', 'car_edate_v', 'n'))
        df_stats['scar_edate_t'] = df_stats.apply(tstat, axis=1, args=('scar_edate_m', 'scar_edate_v', 'scar_edate_n'))

        def tstat2(row, m, s, n):
            try:
                return row[m] / (np_sqrt(row[s]) / row[n])
            except:
                return 0

        df_stats['pat_car'] = df_stats.apply(tstat2, axis=1, args=('scar_m', 'pat_scale_s', 'scar_n'))
        df_stats['pat_car_edate_m'] = df_stats.apply(tstat2, axis=1, args=('scar_edate_m', 'pat_scale_edate_s', 'scar_edate_n'))
        df_stats['pat_ar'] = df_stats.apply(tstat2, axis=1, args=('sar_m', 'pat_scale_s', 'sar_n'))

        # Event window results
        df_evtw = df.loc[(df['isevt'] == 1), ['permno', 'edate', 'rdate', 'evttime', 'ret', 'abret']]
        df_evtw.sort_values(['permno', 'evttime'], ascending=[True, True])

        # Event date results
        maxv = max(levt)
        df_evtd = df.loc[(df['isevt'] == 1) & (df['evttime'] == maxv), ['permno', 'edate', 'cret', 'car', 'bhar']]
        df_evtd.sort_values(['permno', 'edate'], ascending=[True, True])

        # Return results in requested format
        if output == 'df':
            retval = {}
            retval['event_stats'] = df_stats
            retval['event_window'] = df_evtw
            retval['event_date'] = df_evtd
            return retval
        elif output == 'print':
            retval = {}
            print("\nEvent Date")
            print(df_evtd.to_string(index=False))
            print("\n\nEvent Window")
            print(df_evtw.to_string(index=False))
            print("\n\nCross-Sectional Statistics")
            print(df_stats.to_string(index=False))
            return retval
        elif output == 'json':
            retval = {}
            retval['event_stats'] = df_stats.to_dict(orient='split')
            retval['event_window'] = df_evtw.to_dict(orient='split')
            retval['event_date'] = df_evtd.to_dict(orient='split')
            with open(os.path.join(self.output_path, 'EventStudy.json'), 'w') as outfile:
                json_dump(retval, outfile, cls=EncoderJson)
            return json_dumps(retval, cls=EncoderJson)
        elif output == 'csv':
            retval = ''
            es = StringIO_StringIO()
            df_stats.to_csv(es)
            retval += es.getvalue()
            ew = StringIO_StringIO()
            df_evtw.to_csv(ew)
            retval += "\r"
            retval += ew.getvalue()
            ed = StringIO_StringIO()
            df_evtd.to_csv(ed)
            retval += ed.getvalue()
            with open(os.path.join(self.output_path, 'EventStudy.csv'), 'w') as outfile:
                outfile.write(retval)
            return retval
        elif output == 'xls':
            retval = {}
            xlswriter = pd_ExcelWriter(os.path.join(self.output_path, 'EventStudy.xls'))
            df_stats.to_excel(xlswriter, 'Stats')
            df_evtw.to_excel(xlswriter, 'Event Window')
            df_evtd.to_excel(xlswriter, 'Event Date')
            xlswriter.save()
            return retval
        else:
            pass
