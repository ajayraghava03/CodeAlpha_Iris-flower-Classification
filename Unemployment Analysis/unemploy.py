import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILES = [
    'Unemployment_Rate_upto_11_2020.csv',
    'Unemployment in India.csv',
]

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

COLUMN_MAP = {
    'Region': 'Region',
    ' Date': 'Date',
    'Date': 'Date',
    ' Frequency': 'Frequency',
    'Frequency': 'Frequency',
    ' Estimated Unemployment Rate (%)': 'Estimated Unemployment Rate (%)',
    'Estimated Unemployment Rate (%)': 'Estimated Unemployment Rate (%)',
    ' Estimated Employed': 'Estimated Employed',
    'Estimated Employed': 'Estimated Employed',
    ' Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate (%)',
    'Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate (%)',
    'Region.1': 'Region Group',
    'Area': 'Area',
    'longitude': 'Longitude',
    'latitude': 'Latitude',
}

NUMERIC_COLUMNS = [
    'Estimated Unemployment Rate (%)',
    'Estimated Employed',
    'Estimated Labour Participation Rate (%)',
]


def load_unemployment_file(path: str, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={col: COLUMN_MAP.get(col, col.strip()) for col in df.columns})

    df['Source'] = source_name
    df['Region'] = df['Region'].astype(str).str.strip()

    if 'Region Group' in df.columns:
        df['Region Group'] = df['Region Group'].astype(str).str.strip()
    if 'Area' in df.columns:
        df['Area'] = df['Area'].astype(str).str.strip()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_all() -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for filename in DATA_FILES:
        if not os.path.exists(filename):
            print(f"Error: required file '{filename}' not found in the current folder.")
            sys.exit(1)

        source_name = os.path.splitext(filename)[0]
        records.append(load_unemployment_file(filename, source_name))

    combined = pd.concat(records, ignore_index=True, sort=False)
    combined = combined.dropna(subset=['Date', 'Estimated Unemployment Rate (%)'])
    combined['Year'] = combined['Date'].dt.year
    combined['Month'] = combined['Date'].dt.month
    combined['MonthName'] = combined['Date'].dt.strftime('%b')
    return combined


def print_dataset_overview(df: pd.DataFrame) -> None:
    print('Dataset overview:')
    print(f'  Total rows: {len(df):,}')
    print(f'  Date range: {df["Date"].min().date()} to {df["Date"].max().date()}')
    print(f'  Unique regions: {df["Region"].nunique():,}')
    print(f'  Unique sources: {df["Source"].nunique()}\n')

    print('Source coverage:')
    print(df.groupby('Source')['Date'].agg(['min', 'max', 'count']).to_string())
    print('\nMost recent unemployment values by source:')
    print(
        df.sort_values(['Source', 'Date'], ascending=[True, False])
        .groupby('Source')
        .head(1)
        .loc[:, ['Source', 'Date', 'Region', 'Estimated Unemployment Rate (%)']]
        .to_string(index=False)
    )
    print()


def plot_time_series(df: pd.DataFrame, path: str, title: str, legend: bool = False) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(df['Date'], df['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Estimated Unemployment Rate (%)')
    if legend:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def main() -> None:
    df = load_all()
    print_dataset_overview(df)

    national = (
        df.groupby('Date', as_index=False)['Estimated Unemployment Rate (%)']
        .mean()
        .sort_values('Date')
    )

    plot_time_series(national, os.path.join(PLOT_DIR, 'unemployment_trend.png'), 'National Average Unemployment Rate Over Time')

    covid_start = pd.Timestamp('2020-03-01')
    pre_covid = national[national['Date'] < covid_start]
    covid_period = national[national['Date'] >= covid_start]

    pre_avg = pre_covid['Estimated Unemployment Rate (%)'].mean()
    covid_avg = covid_period['Estimated Unemployment Rate (%)'].mean()
    change_pct = ((covid_avg - pre_avg) / pre_avg * 100) if pre_avg else float('nan')

    print('COVID-19 impact summary:')
    print(f'  Pre-COVID average (before Mar 2020): {pre_avg:.2f}%')
    print(f'  COVID-period average (Mar 2020 onwards): {covid_avg:.2f}%')
    print(f'  Average change: {change_pct:+.2f}%\n')

    plt.figure(figsize=(11, 5))
    plt.plot(pre_covid['Date'], pre_covid['Estimated Unemployment Rate (%)'], marker='o', label='Pre-COVID')
    plt.plot(covid_period['Date'], covid_period['Estimated Unemployment Rate (%)'], marker='o', label='COVID period')
    plt.title('COVID-19 Impact on National Unemployment Rate')
    plt.xlabel('Date')
    plt.ylabel('Estimated Unemployment Rate (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    covid_path = os.path.join(PLOT_DIR, 'covid_impact.png')
    plt.savefig(covid_path, dpi=150)
    plt.close()
    print(f"Saved plot: {covid_path}")

    monthly = (
        df.groupby('Month', as_index=False)['Estimated Unemployment Rate (%)']
        .mean()
        .sort_values('Month')
    )
    monthly['MonthName'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(11, 5))
    plt.plot(monthly['MonthName'], monthly['Estimated Unemployment Rate (%)'], marker='o')
    plt.title('Seasonal Unemployment Pattern by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Estimated Unemployment Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    season_path = os.path.join(PLOT_DIR, 'seasonal_pattern.png')
    plt.savefig(season_path, dpi=150)
    plt.close()
    print(f"Saved plot: {season_path}\n")

    region_avg = (
        df.groupby('Region')['Estimated Unemployment Rate (%)']
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )
    print('Top regions by average unemployment rate:')
    for region, rate in region_avg.items():
        print(f'  {region}: {rate:.2f}%')
    print()

    if 'Area' in df.columns:
        area_avg = (
            df.groupby('Area')['Estimated Unemployment Rate (%)']
            .mean()
            .sort_values(ascending=False)
        )
        print('Average unemployment rate by area:')
        print(area_avg.to_string())
        print()

    if 'Region Group' in df.columns:
        group_avg = (
            df.groupby('Region Group')['Estimated Unemployment Rate (%)']
            .mean()
            .sort_values(ascending=False)
        )
        print('Average unemployment rate by region group:')
        print(group_avg.to_string())
        print()

    feb_2020 = df[df['Date'].dt.to_period('M') == pd.Period('2020-02', 'M')]
    apr_2020 = df[df['Date'].dt.to_period('M') == pd.Period('2020-04', 'M')]
    if not feb_2020.empty and not apr_2020.empty:
        spike = (
            apr_2020.groupby('Region')['Estimated Unemployment Rate (%)'].mean() -
            feb_2020.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
        ).sort_values(ascending=False).head(6)
        print('Largest regional unemployment spikes from Feb 2020 to Apr 2020:')
        for region, diff in spike.items():
            print(f'  {region}: {diff:.2f}%')
        print()

    print('Key insights:')
    print('  - Combining both datasets gives a broader national and regional view of unemployment trends.')
    print('  - The national unemployment rate increased significantly after March 2020, reflecting COVID-19 disruption.')
    print('  - Monthly seasonality highlights periods with systematically higher unemployment rates.')
    print('  - The highest average rates and COVID spikes identify regions that may need focused policy support.')
    print('  - If area data exists, rural and urban differences can guide social intervention planning.')


if __name__ == '__main__':
    main()
