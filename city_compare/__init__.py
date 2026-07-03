"""
city_compare -- China City Economic Comparison Dashboard
========================================================
多维度城市经济对比工具包。

Modules:
    fetcher  -- 数据获取 (akshare + YAML 回退)
    radar    -- 可视化引擎 (Plotly)
    report   -- HTML 报告生成器
"""

__version__ = "2.0.0"
__author__ = "city-compare"

from city_compare.fetcher import (
    compare_cities,
    compute_composite_scores,
    get_city_gdp,
    get_city_budget,
    get_city_population,
    get_city_income,
    get_city_retail,
    get_city_industry,
    get_city_fdi,
    get_city_education,
    get_city_healthcare,
    load_city_config,
    list_available_cities,
)

from city_compare.radar import (
    build_radar,
    build_bar_race,
    build_correlation_matrix,
    build_city_clustering,
    build_gap_analysis,
    build_rank_table,
    COLORS,
    DIMENSION_LABELS,
)

from city_compare.report import generate_report
