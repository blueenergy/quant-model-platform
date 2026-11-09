"""All-A (Shanghai/Shenzhen full-market) production pipeline.

Panel build -> size-neutral label -> lightgbm score -> top20 industry-neutral
selection -> paper ledger. No index-membership dependency: the universe is every
SH/SZ stock with a price row and >= 20M CNY 20-day average turnover.
"""
