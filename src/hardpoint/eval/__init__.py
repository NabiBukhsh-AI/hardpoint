"""Datasets, the eval runner, metrics, baselines, and reports.

Owns evaluation plumbing. Does not own the user's golden data, and never builds a
second execution path: the runner drives the production Pipeline object.

Populated in milestone M3.
"""
