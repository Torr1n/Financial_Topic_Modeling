import wrds
import pandas as pd
import math
from event_study_module import EventStudy
import os
from pathlib import Path
import numpy as np
import statsmodels.api as sm

class ThematicES:

     #Initialize with dictionary attribute. Dictionary must have three dimensions: permno, edate, and sentiment. e.g.:
     # events = [{"permno":10002,"edate":"05/29/2012","sentiment": 1},
     #       {"permno":82504,"edate":"05/29/2012","sentiment": 2},
     #       {"permno":89350,"edate":"01/04/2010","sentiment": 3},
     #       {"permno":14593, "edate":"01/04/2022","sentiment": 4}]

     def __init__(self, dictionary, wrds_connection=None):
          self.dictionary = dictionary
          self.WRDSQuery = None
          self.CRSPQuery = None
          self.Factors = None
          self.Results = None
          self.wrds_connection = wrds_connection
          self.owns_connection = False  # Track if we created the connection

     def wrdsPull(self):
          #Extract permnos from dictionary
          permnos = [event["permno"] for event in self.dictionary]

          #Convert to list for sql querying
          permno_str = ",".join(str(p) for p in permnos)

          # Use existing connection or create new one
          if self.wrds_connection is not None:
               conn = self.wrds_connection
          else:
               conn = wrds.Connection()
               self.owns_connection = True  # We created it, so we'll close it

          #Extract WRDS data using SQL query for each permno
          self.WRDSQuery = conn.raw_sql(f"""WITH funda AS (
          SELECT
               gvkey,
               datadate,
               SALE,
               EMP AS employees,
               datafmt,
               consol,
               indfmt
          FROM comp_na_daily_all.funda
          WHERE datafmt = 'STD'
               AND consol  = 'C'
          ),
          compustat_na AS (
          SELECT DISTINCT
               fq.gvkey,
               fq.datadate,
               fq.fyearq,
               fq.fqtr,
               fq.fyr,
               fq.tic,
               fq.EPSFXQ AS eps_q,
               fq.IBQ AS net_income_q,
               fq.ATQ AS total_assets_q,
               fq.DLTTQ AS total_debt_q,
               fq.LCTQ AS current_liabilities_total_q,
               fq.TEQQ AS shareholders_equity_q,
               fq.CAPXY AS capex_q,
               fl.SALE,
               fq.CHEQ AS cash_and_short_term_investments_q,
               fq.PPENTQ AS ppe_q,
               fl.employees,
               fq.XRDQ AS research_and_development_expense_q
          FROM comp_na_daily_all.fundq AS fq
          LEFT JOIN funda fl
               ON fq.gvkey = fl.gvkey
               AND fq.datadate = fl.datadate
               AND fq.datafmt = fl.datafmt
               AND fq.consol = fl.consol
               AND fq.indfmt = fl.indfmt
               WHERE fq.datafmt = 'STD'
               AND fq.consol = 'C'
          ),
          ibes_ranked AS (
          SELECT
               i.ticker,
               i.fpedats,
               i.statpers,
               i.numest,
               i.meanest,
               i.medest,
               i.measure,
               i.actual,
               ROW_NUMBER() OVER (
               PARTITION BY i.ticker, i.fpedats
               ORDER BY i.statpers DESC
               ) AS rn
               FROM tr_ibes.statsum_epsus AS i
               WHERE i.measure = 'EPS'
               AND i.fiscalp = 'QTR'
          ),
          ibes_latest AS (
          SELECT 
               *
          FROM ibes_ranked
          WHERE rn = 1
          ),
          final_joined AS (
          SELECT DISTINCT
               s.permno,
               s.ticker,
               s.comnam,
               c.*,
               ib.measure,
               ib.fpedats,
               ib.meanest,
               ib.medest,
               ib.actual
          FROM compustat_na c
          JOIN crsp.ccmxpf_linktable l
               ON c.gvkey = l.gvkey
               AND c.datadate BETWEEN l.linkdt AND COALESCE(l.linkenddt, '9999-12-31')
          JOIN crsp.stocknames s
               ON s.permno = l.lpermno
               AND c.datadate BETWEEN s.namedt AND COALESCE(s.nameenddt, '9999-12-31')
          LEFT JOIN wrdsapps_link_crsp_ibes.ibcrsphist ib_ln
               ON s.permno = ib_ln.permno
               AND c.datadate BETWEEN ib_ln.sdate AND COALESCE(ib_ln.edate, '9999-12-31')
          LEFT JOIN ibes_latest ib
          ON ib_ln.ticker   = ib.ticker
          AND c.datadate = ib.fpedats
          AND ib.measure = 'EPS'
          WHERE l.lpermno IN ({permno_str})
               AND l.linktype IN ('LU', 'LC')
               AND l.linkprim IN ('P', 'C')
          )

          SELECT *
          FROM final_joined;

          """)

          self.CRSPQuery = conn.raw_sql(f"""
          SELECT
               *
          FROM crsp_a_stock.msf
          WHERE msf.permno IN ({permno_str})""")

          #Preliminary calculations before merging
          self.CRSPQuery["prc"] = self.CRSPQuery["prc"].abs()
          self.CRSPQuery["cap"] = self.CRSPQuery["prc"] * self.CRSPQuery["shrout"]
          self.CRSPQuery["ret_sqd"] = self.CRSPQuery["ret"]**2
          self.CRSPQuery["gross_ret"] = self.CRSPQuery["ret"]+1
          #Compute stock volatility using sqrt(cumulative sum of 12 months of squared returns) and stock return (buy and hold return, or cumulative product of last 3 months gross returns-1)
          self.CRSPQuery = self.CRSPQuery.sort_values(["permno", "date"])
          self.CRSPQuery["Stock_Volatility"] = np.sqrt(self.CRSPQuery.groupby(['permno','cusip'])["ret_sqd"].transform(lambda s: s.rolling(window=12, min_periods = 12).sum()))
          self.CRSPQuery["Stock_Return"] = self.CRSPQuery.groupby(['permno','cusip'])["gross_ret"].transform(lambda s: s.rolling(window=3, min_periods = 3).apply(np.prod, raw=True))-1

          #Prepare factors table and CAR table for merging
          self.WRDSQuery['datadate'] = pd.to_datetime(self.WRDSQuery['datadate'], format='%Y-%m-%d')
          self.CRSPQuery['date'] = pd.to_datetime(self.CRSPQuery['date'], format='%Y-%m-%d')

          #Sort dates in both tables for merging
          self.WRDSQuery = self.WRDSQuery.sort_values(['datadate'])
          self.CRSPQuery = self.CRSPQuery.sort_values(['date'])

          #Merge CRSPQuery and WRDSQuery tables using fuzzy merge (backwards date match, give or take 31 days of tolerance). This is because CRSP is using month-end data, which may or may not correspond with the date of fiscal quarter-ends. 31 days is a large tolerance, however, and could lead to issues with dated return data
          self.Factors = pd.merge_asof(
          self.WRDSQuery,
          self.CRSPQuery,
          by="permno",                         # exact match on permno
          left_on="datadate",
          right_on="date",
          direction="backward",                  # nearest date, not strictly backwards
          tolerance=pd.Timedelta("31D")          # allow up to 4-day difference
          )
          
          #Close the connection
          # Only close if we created the connection
          if self.owns_connection:
               conn.close()
               self.owns_connection = False

     def calculateFactors(self):
     
          # Ensure correct sort for forward-fill
          self.Factors = self.Factors.sort_values(['permno', 'ticker', 'datadate'])

          # Forward-fill annual figures within each (permno, ticker), with a limit of 3 consecutive NAs (only ffill relevant annual results)
          self.Factors['sale'] = self.Factors.groupby(['permno', 'ticker'], sort=False)['sale'].ffill(limit = 3)
          self.Factors['employees'] = self.Factors.groupby(['permno', 'ticker'], sort=False)['employees'].ffill(limit = 3)

          self.Factors = self.Factors.sort_values(['permno','ticker','fyearq','fqtr'])

          #Lag columns
          self.Factors['sale_lagged'] = self.Factors.groupby(['permno','ticker','fqtr'])['sale'].shift(1)
          self.Factors['employees_lagged'] = self.Factors.groupby(['permno','ticker','fqtr'])['employees'].shift(1)

          #Compute factors from Labor-Shortage Exposure Paper
          self.Factors['Return_on_Assets'] = self.Factors['net_income_q'] / self.Factors['total_assets_q']
          self.Factors['Book_Leverage'] = (self.Factors['total_debt_q'] + self.Factors['current_liabilities_total_q']) / (self.Factors['total_debt_q'] + self.Factors['current_liabilities_total_q']+self.Factors['shareholders_equity_q'])
          self.Factors['Capital_Expenditures'] = self.Factors['capex_q'] / self.Factors['total_assets_q']
          self.Factors['Research_and_Development'] = self.Factors['research_and_development_expense_q'] / self.Factors['total_assets_q']
          self.Factors['Sales_Growth'] = self.Factors['sale'] / self.Factors['sale_lagged']
          
          # Handle log transformation safely - replace zero/negative/missing sales with NaN
          # Use pandas method to handle NaN values properly
          self.Factors['Firm_Size'] = self.Factors['sale'].apply(
              lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan
          )
          
          self.Factors['Cash'] = self.Factors['cash_and_short_term_investments_q'] / self.Factors['total_assets_q']
          self.Factors['Asset_Tangibility'] = self.Factors['ppe_q'] / self.Factors['total_assets_q']

          #Market value of assets is calculated as market cap + total debt - cash (Enterprise Value)
          self.Factors['Market_to_Book'] = (self.Factors['cap']+self.Factors['total_debt_q']-self.Factors['cash_and_short_term_investments_q'])/self.Factors['total_assets_q']
          self.Factors['Earnings_Surprise'] = (self.Factors['actual'] - self.Factors['medest'])*self.Factors['prc']

          #Should this be Δ(employees) / total assets OR Δ(employees/total assets)? For now, implementing former
          self.Factors['Delta_Employee_Change'] = (self.Factors['employees'] - self.Factors['employees_lagged']) / self.Factors['total_assets_q']

          #To do: Stock return, MTB, Stock volatility, earnings surprise

     def calculateCovariatesAndCAR(self):
          """
          Calculate covariates and CAR for all events.
          Returns a DataFrame with all covariates, CAR, and sentiment ready for regression.
          Does NOT run regression - that should be done at the pipeline level for per-theme analysis.
          """

          #Initialize event study class with shared connection
          eventstudy = EventStudy(
               output_path=os.path.join(Path.home()),
               wrds_connection=self.wrds_connection
          )

          #Reformat dictionary data for event study
          events = [{"permno": e["permno"], "edate": e["edate"]} for e in self.dictionary]

          #Run event study
          result = eventstudy.eventstudy(data=events, model='madj', output='df')

          #Keep df that fits format for analysis
          car = result.get("event_date")

          #Prepare factors table and CAR table for merging
          self.Factors['datadate'] = pd.to_datetime(self.Factors['datadate'], format='%Y-%m-%d')
          car['edate'] = pd.to_datetime(car['edate'], format='%Y-%m-%d')

          #Sort dates in both tables for merging
          self.Factors = self.Factors.sort_values(['datadate'])
          car = car.sort_values(['edate'])

          self.Results = pd.merge_asof(car, self.Factors, left_on = 'edate', right_on ='datadate', by = 'permno', direction = 'backward', allow_exact_matches= False)

          self.Results = self.Results[['permno', 'edate', 'datadate', 'ticker', 'comnam', 'Return_on_Assets', 'Book_Leverage', 'Capital_Expenditures', 'Research_and_Development', 'Sales_Growth', 'Firm_Size', 'Cash', 'Asset_Tangibility', 'Delta_Employee_Change','Stock_Volatility','Stock_Return','Market_to_Book', 'Earnings_Surprise','cret', 'car', 'bhar']]

          #Drop rows / events with NA values
          self.Results = self.Results.dropna()

          # Also drop rows with infinite values in any column
          self.Results = self.Results.replace([np.inf, -np.inf], np.nan).dropna()

          #Convert dictionary to table and adjust edate type to allow for merging
          dictionary_table = pd.DataFrame(self.dictionary)
          dictionary_table["edate"] = pd.to_datetime(dictionary_table["edate"])

          self.Results = self.Results.merge(dictionary_table, on=["permno", "edate"], how="left")

          # Return results DataFrame (regression will be done at pipeline level)
          return self.Results

     def doAll(self):
          """
          Run complete event study analysis: pull WRDS data, calculate factors, and compute CAR.
          Returns DataFrame with all covariates, CAR, and sentiment ready for regression.
          """
          self.wrdsPull()
          self.calculateFactors()
          return self.calculateCovariatesAndCAR()