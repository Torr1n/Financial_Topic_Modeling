import pandas as pd
import numpy as np

from src.wrds_connection import WRDSConnection


class PortfolioSorts():
    """
    Portfolio sorts analysis for thematic sentiment studies.

    This class computes portfolio returns by sorting securities into sentiment
    terciles (Low, Medium, High) and tracking cumulative returns over time.

    Args:
        dictionary: List of event dictionaries with keys: permno, edate, sentiment.
            Example: [{"permno": 10002, "edate": "05/29/2012", "sentiment": 1}, ...]
        wrds_connection: Optional existing WRDS connection. If provided, the
            connection will be reused and NOT closed by this class.
        weighting: Portfolio weighting method - "value" (default) or "equal".
    """

    def __init__(self, dictionary, wrds_connection=None, weighting: str = "value"):
        """
        Initialize the portfolio sorts analysis.

        Args:
            dictionary: List of event dictionaries with permno, edate, sentiment.
            wrds_connection: Optional existing WRDS connection to reuse.
            weighting: Portfolio weighting method - "value" or "equal".
        """
        self.dictionary = dictionary
        self.CRSPQuery = None
        self.wrds_connection = wrds_connection
        self.portfolio_returns = None
        self.weighting = weighting

    def crspreturns(self):
        """
        Pull CRSP daily stock returns for all events.

        Uses WRDSConnection context manager to properly manage connection lifecycle.
        If an external connection was provided, it will be reused and not closed.
        """
        # Extract permnos and edates from dictionary
        permnos = [event["permno"] for event in self.dictionary]
        permno_list = ','.join(str(p) for p in permnos)

        # Derive date range from event dates (dynamic, not hardcoded)
        edates = [pd.to_datetime(event["edate"]) for event in self.dictionary]
        min_edate = min(edates)
        max_edate = max(edates)

        # Start from earliest event, end 120 days after latest event
        start_date = min_edate.strftime('%Y-%m-%d')
        end_date = (max_edate + pd.Timedelta(days=120)).strftime('%Y-%m-%d')

        print(f"Fetching CRSP data for {len(permnos)} firms from {start_date} to {end_date}")

        # Use context manager for connection management
        with WRDSConnection(connection=self.wrds_connection) as conn:
            # Simple, efficient query using IN clause and fixed date range
            self.CRSPQuery = conn.raw_sql(f"""
            SELECT
                permno,
                date,
                prc,
                ret,
                shrout,
                vol,
                cfacpr,
                cfacshr
            FROM crsp_a_stock.dsf
            WHERE permno IN ({permno_list})
                AND date >= '{start_date}'::date
                AND date <= '{end_date}'::date
            ORDER BY permno, date""")

        self.CRSPQuery["date"] = pd.to_datetime(self.CRSPQuery['date'], format='%Y-%m-%d')

        # Convert dictionary to table and adjust edate type to allow for merging
        dictionary_table = pd.DataFrame(self.dictionary)
        dictionary_table["edate"] = pd.to_datetime(dictionary_table["edate"])

        self.CRSPQuery = self.CRSPQuery.merge(dictionary_table, on=["permno"], how="left")

        # Filter to only include data from edate onwards for each permno
        print(f"  Before filtering to edate: {len(self.CRSPQuery)} rows")
        self.CRSPQuery = self.CRSPQuery[self.CRSPQuery['date'] >= self.CRSPQuery['edate']]
        print(f"  After filtering to edate onwards: {len(self.CRSPQuery)} rows")

        # Calculate trading days from event (0-indexed, where event date = 0)
        self.CRSPQuery = self.CRSPQuery.sort_values(['permno', 'date'])
        self.CRSPQuery['days_from_event'] = self.CRSPQuery.groupby('permno').cumcount()

    def compute_portfolio_returns(self):
        """
        Split securities into 3 sentiment buckets and compute cumulative returns.

        Returns:
            DataFrame with portfolio returns by sentiment bucket and day.
        """
        if self.CRSPQuery is None or self.CRSPQuery.empty:
            print("No CRSP data available. Run crspreturns() first.")
            return None

        # Calculate market cap for value weighting (price * shares outstanding)
        self.CRSPQuery['mktcap'] = abs(self.CRSPQuery['prc']) * self.CRSPQuery['shrout']

        # Create sentiment terciles (Low, Medium, High)
        sentiment_percentiles = self.CRSPQuery.groupby('days_from_event')['sentiment'].quantile([0.33, 0.67]).unstack()

        # Assign securities to buckets based on sentiment
        def assign_bucket(row):
            try:
                if row['days_from_event'] in sentiment_percentiles.index:
                    day_percentiles = sentiment_percentiles.loc[row['days_from_event']]
                    if row['sentiment'] <= day_percentiles[0.33]:
                        return 'Low'
                    elif row['sentiment'] <= day_percentiles[0.67]:
                        return 'Medium'
                    else:
                        return 'High'
                else:
                    # Fallback to overall percentiles if day not found
                    overall_33 = self.CRSPQuery['sentiment'].quantile(0.33)
                    overall_67 = self.CRSPQuery['sentiment'].quantile(0.67)
                    if row['sentiment'] <= overall_33:
                        return 'Low'
                    elif row['sentiment'] <= overall_67:
                        return 'Medium'
                    else:
                        return 'High'
            except Exception:
                return 'Medium'

        self.CRSPQuery['sentiment_bucket'] = self.CRSPQuery.apply(assign_bucket, axis=1)

        # Calculate value-weighted returns for each bucket and day
        portfolio_returns = []

        for bucket in ['Low', 'Medium', 'High']:
            bucket_data = self.CRSPQuery[self.CRSPQuery['sentiment_bucket'] == bucket].copy()

            # Group by days_from_event and calculate value-weighted return
            # Limit to maximum 90 days from event
            daily_returns = []
            unique_days = sorted(bucket_data['days_from_event'].unique())
            unique_days = [d for d in unique_days if 0 <= d <= 90]

            for day in unique_days:
                day_data = bucket_data[bucket_data['days_from_event'] == day].copy()

                if day_data.empty:
                    continue

                # Calculate weights based on weighting method
                if self.weighting == "equal":
                    vw_return = day_data['ret'].mean() if len(day_data) > 0 else 0
                else:
                    # Value-weighted return (default)
                    total_mktcap = day_data['mktcap'].sum()
                    if total_mktcap > 0:
                        weights = day_data['mktcap'].values / total_mktcap
                        vw_return = (day_data['ret'].values * weights).sum()
                    else:
                        vw_return = 0

                daily_returns.append({
                    'bucket': bucket,
                    'days_from_event': day,
                    'vw_return': vw_return
                })

            # Convert to DataFrame and calculate cumulative returns
            if not daily_returns:
                # No data for this bucket, skip
                continue

            bucket_returns = pd.DataFrame(daily_returns)
            bucket_returns = bucket_returns.sort_values('days_from_event')

            # Ensure day 0 exists
            if 0 not in bucket_returns['days_from_event'].values:
                day0_row = pd.DataFrame([{'bucket': bucket, 'days_from_event': 0, 'vw_return': 0}])
                bucket_returns = pd.concat([day0_row, bucket_returns]).sort_values('days_from_event')

            # Calculate cumulative returns starting from day 1
            bucket_returns['cumulative_return'] = 0.0
            for i in range(len(bucket_returns)):
                if bucket_returns.iloc[i]['days_from_event'] == 0:
                    bucket_returns.loc[bucket_returns.index[i], 'cumulative_return'] = 0.0
                else:
                    mask = (bucket_returns['days_from_event'] > 0) & (bucket_returns['days_from_event'] <= bucket_returns.iloc[i]['days_from_event'])
                    daily_rets = bucket_returns.loc[mask, 'vw_return'].values
                    bucket_returns.loc[bucket_returns.index[i], 'cumulative_return'] = (np.prod(1 + daily_rets) - 1) if len(daily_rets) > 0 else 0

            # Verify cumulative calculation (for debugging)
            if len(bucket_returns) > 0:
                for i in range(min(3, len(bucket_returns))):
                    row = bucket_returns.iloc[i]
                    if i == 0:
                        expected = row['vw_return']
                    else:
                        prev_cum = bucket_returns.iloc[i-1]['cumulative_return']
                        expected = (1 + prev_cum) * (1 + row['vw_return']) - 1

                    actual = row['cumulative_return']
                    if abs(actual - expected) > 0.0001:
                        print(f"WARNING: Cumulative return mismatch for {bucket} day {row['days_from_event']}")
                        print(f"  Expected: {expected:.6f}, Actual: {actual:.6f}")

            portfolio_returns.append(bucket_returns)

        # Combine all buckets
        self.portfolio_returns = pd.concat(portfolio_returns, ignore_index=True)

        # Check for duplicate day/bucket combinations
        duplicates = self.portfolio_returns.groupby(['bucket', 'days_from_event']).size()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            print("WARNING: Found duplicate day/bucket combinations:")
            print(duplicates)

        # Print summary
        print("\nPortfolio Returns by Sentiment Bucket:")
        print("=" * 60)
        for bucket in ['Low', 'Medium', 'High']:
            bucket_data = self.portfolio_returns[self.portfolio_returns['bucket'] == bucket]
            if not bucket_data.empty:
                max_day = bucket_data['days_from_event'].max()
                final_data = bucket_data[bucket_data['days_from_event'] == max_day]['cumulative_return']
                if len(final_data) > 0:
                    final_return = final_data.values[0]
                    print(f"{bucket} Sentiment: {final_return:.4%} cumulative return")
                else:
                    print(f"{bucket} Sentiment: No data for max day")

        return self.portfolio_returns
