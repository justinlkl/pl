#!/usr/bin/env python3
"""Inspect fixture data structure."""

import pandas as pd

# Read fixtures from GW1
fixtures = pd.read_csv('FPL-Core-Insights/data/2025-2026/By Gameweek/GW1/fixtures.csv')
print("Fixtures columns (first 20):", fixtures.columns.tolist()[:20])
print("\nFirst 3 fixtures:")
print(fixtures[['gameweek', 'home_team', 'away_team']].head(3))
print("\nSample fixture:")
print(fixtures.iloc[0][['gameweek', 'home_team', 'away_team', 'home_score', 'away_score']])

# Check team codes
teams = pd.read_csv('FPL-Core-Insights/data/2025-2026/teams.csv')
print("\n\nTeams columns:", teams.columns.tolist())
print("\nTeams data (first 5):")
print(teams.head(5))
