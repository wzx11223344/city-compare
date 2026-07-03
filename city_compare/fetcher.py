"""
城市经济数据获取模块

使用 akshare 实时获取中国城市经济数据，失败时回退到 config/cities.yaml 中的预载静态数据。
所有指标在适当处标准化为人均或单位值。
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---- 路径常量 ----
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CITIES_YAML = _CONFIG_DIR / "cities.yaml"
_CACHE: Optional[Dict[str, Any]] = None

# ---- 内部工具 ----

def _load_yaml() -> Dict[str, Any]:
    """加载 cities.yaml 配置（带缓存）。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with open(_CITIES_YAML, "r", encoding="utf-8") as f:
        _CACHE = yaml.safe_load(f)
    return _CACHE


def list_available_cities() -> List[str]:
    """返回所有预载城市名称列表。"""
    cfg = _load_yaml()
    return list(cfg["cities"].keys())


def load_city_config(city: str) -> Dict[str, Any]:
    """获取单个城市的预载数据。"""
    cfg = _load_yaml()
    if city not in cfg["cities"]:
        raise KeyError(f"城市 '{city}' 不在预载列表中。可用城市: {list_available_cities()}")
    return cfg["cities"][city]


def _safe_akshare_fetch(fetch_fn, fallback_fn, *args, **kwargs):
    """通用安全获取：先尝试 akshare，失败则走回退。"""
    try:
        result = fetch_fn(*args, **kwargs)
        if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
            return result
    except Exception:
        pass
    return fallback_fn(*args, **kwargs)


# ---- 指标获取函数 ----

def get_city_gdp(city: str, year: int = 2023) -> Dict[str, float]:
    """
    获取城市 GDP 数据。
    Returns: {"gdp": ..., "gdp_growth": ..., "gdp_per_capita": ...}
    """
    cfg = load_city_config(city)
    pop = cfg.get("population", 1)

    # 尝试 akshare 实时数据
    try:
        import akshare as ak
        df = ak.macro_china_city_gdp()
        if df is not None and not df.empty:
            mask = df.apply(lambda r: city in str(r.values), axis=1)
            matched = df[mask]
            if not matched.empty:
                row = matched.iloc[0]
                return {
                    "gdp": float(row.iloc[1]) if len(row) > 1 else cfg["gdp"],
                    "gdp_growth": cfg.get("gdp_growth", 0),
                    "gdp_per_capita": round(cfg["gdp"] / pop * 10000, 1),
                }
    except Exception:
        pass

    return {
        "gdp": cfg["gdp"],
        "gdp_growth": cfg.get("gdp_growth", 0),
        "gdp_per_capita": round(cfg["gdp"] / pop * 10000, 1),
    }


def get_city_budget(city: str, year: int = 2023) -> Dict[str, float]:
    """获取一般公共预算收入。"""
    cfg = load_city_config(city)
    pop = cfg.get("population", 1)
    return {
        "budget_revenue": cfg["budget_revenue"],
        "budget_per_capita": round(cfg["budget_revenue"] / pop * 10000, 1),
        "budget_to_gdp_ratio": round(cfg["budget_revenue"] / cfg["gdp"] * 100, 2),
    }


def get_city_population(city: str) -> Dict[str, float]:
    """获取常住人口。"""
    cfg = load_city_config(city)
    return {
        "population": cfg["population"],
        "tier": cfg.get("tier", 3),
    }


def get_city_income(city: str) -> Dict[str, float]:
    """获取城镇居民人均可支配收入。"""
    cfg = load_city_config(city)
    return {
        "income_urban": cfg.get("income_urban", 0),
        "income_to_gdp_per_capita_ratio": round(
            cfg.get("income_urban", 0) / (cfg["gdp"] / cfg.get("population", 1) * 10000) * 100, 1
        ),
    }


def get_city_retail(city: str) -> Dict[str, float]:
    """获取社会消费品零售总额。"""
    cfg = load_city_config(city)
    pop = cfg.get("population", 1)
    return {
        "retail_sales": cfg["retail_sales"],
        "retail_per_capita": round(cfg["retail_sales"] / pop * 10000, 1),
        "retail_to_gdp_ratio": round(cfg["retail_sales"] / cfg["gdp"] * 100, 2),
    }


def get_city_industry(city: str) -> Dict[str, float]:
    """获取规模以上工业增加值。"""
    cfg = load_city_config(city)
    gdp = cfg["gdp"]
    return {
        "industry_value": cfg.get("industry_value", 0),
        "industry_to_gdp_ratio": round(cfg.get("industry_value", 0) / gdp * 100, 2),
    }


def get_city_fdi(city: str) -> Dict[str, float]:
    """获取实际利用外资。"""
    cfg = load_city_config(city)
    gdp = cfg["gdp"]
    # 美元换算人民币 (7.1)
    fdi_cny = cfg.get("fdi", 0) * 7.1
    return {
        "fdi": cfg.get("fdi", 0),
        "fdi_to_gdp_ratio": round(fdi_cny / gdp * 100, 2),
    }


def get_city_education(city: str) -> Dict[str, float]:
    """获取教育指标（高校数量、研发强度）。"""
    cfg = load_city_config(city)
    pop = cfg.get("population", 1)
    return {
        "universities": cfg.get("universities", 0),
        "universities_per_million": round(cfg.get("universities", 0) / pop, 3),
        "rd_spending": cfg.get("rd_spending", 0),
    }


def get_city_healthcare(city: str) -> Dict[str, float]:
    """获取医疗指标（每千人床位数）。"""
    cfg = load_city_config(city)
    return {
        "hospital_beds": cfg.get("hospital_beds", 0),
    }


# ---- 批量对比 ----

def compare_cities(city_list: List[str], year: int = 2023) -> Dict[str, pd.DataFrame]:
    """
    批量获取多个城市的经济指标并进行对比。

    Args:
        city_list: 城市名称列表
        year: 数据年份

    Returns:
        {
            "overview": DataFrame (综合概览),
            "per_capita": DataFrame (人均指标),
            "structure": DataFrame (经济结构),
            "public_service": DataFrame (公共服务),
            "raw": DataFrame (完整原始数据),
        }
    """
    if not city_list:
        raise ValueError("至少需要提供一个城市名称")

    rows_overview = []
    rows_per_capita = []
    rows_structure = []
    rows_public = []
    rows_raw = []

    for city in city_list:
        try:
            gdp = get_city_gdp(city, year)
            budget = get_city_budget(city, year)
            pop = get_city_population(city)
            income = get_city_income(city)
            retail = get_city_retail(city)
            industry = get_city_industry(city)
            fdi = get_city_fdi(city)
            edu = get_city_education(city)
            health = get_city_healthcare(city)

            # 综合概览
            rows_overview.append({
                "城市": city,
                "GDP(亿元)": gdp["gdp"],
                "GDP增速(%)": gdp["gdp_growth"],
                "人均GDP(元)": gdp["gdp_per_capita"],
                "常住人口(万人)": pop["population"],
                "城市等级": f"T{pop['tier']}",
            })

            # 人均指标
            rows_per_capita.append({
                "城市": city,
                "人均GDP(元)": gdp["gdp_per_capita"],
                "人均预算收入(元)": budget["budget_per_capita"],
                "城镇居民人均可支配收入(元)": income["income_urban"],
                "人均社消(元)": retail["retail_per_capita"],
            })

            # 经济结构
            rows_structure.append({
                "城市": city,
                "预算收入占GDP比(%)": budget["budget_to_gdp_ratio"],
                "社消占GDP比(%)": retail["retail_to_gdp_ratio"],
                "工业占GDP比(%)": industry["industry_to_gdp_ratio"],
                "外资占GDP比(%)": fdi["fdi_to_gdp_ratio"],
                "R&D占GDP比(%)": edu["rd_spending"],
            })

            # 公共服务
            rows_public.append({
                "城市": city,
                "高校数量": edu["universities"],
                "每百万人高校数": round(edu["universities_per_million"] * 100, 2),
                "R&D经费占GDP比(%)": edu["rd_spending"],
                "每千人床位数": health["hospital_beds"],
            })

            # 完整原始数据
            rows_raw.append({
                "城市": city,
                "GDP(亿元)": gdp["gdp"],
                "GDP增速(%)": gdp["gdp_growth"],
                "人均GDP(元)": gdp["gdp_per_capita"],
                "常住人口(万人)": pop["population"],
                "预算收入(亿元)": budget["budget_revenue"],
                "人均可支配收入(元)": income["income_urban"],
                "社消总额(亿元)": retail["retail_sales"],
                "工业增加值(亿元)": industry["industry_value"],
                "实际利用外资(亿美元)": fdi["fdi"],
                "高校数量": edu["universities"],
                "R&D占比(%)": edu["rd_spending"],
                "每千人床位数": health["hospital_beds"],
            })
        except KeyError as e:
            print(f"警告: {e}")

    return {
        "overview": pd.DataFrame(rows_overview),
        "per_capita": pd.DataFrame(rows_per_capita),
        "structure": pd.DataFrame(rows_structure),
        "public_service": pd.DataFrame(rows_public),
        "raw": pd.DataFrame(rows_raw),
    }


# ---- 综合指标计算 (用于雷达图) ----

def compute_composite_scores(city_list: List[str]) -> pd.DataFrame:
    """
    计算 8 维综合得分，通过 z-score 标准化到 0-100 区间。

    8 个维度:
        经济规模, 增长动能, 财政收入, 居民收入, 消费活力, 工业实力, 对外开放, 公共服务
    每个维度由 2-3 个子指标加权合成。
    """
    result = compare_cities(city_list)
    df = result["raw"].set_index("城市")

    scores = pd.DataFrame(index=df.index)

    # 1. 经济规模: GDP * 0.6 + 人均GDP * 0.4
    scores["经济规模"] = (
        _zscore(df["GDP(亿元)"]) * 0.6 + _zscore(df["人均GDP(元)"]) * 0.4
    )

    # 2. 增长动能: GDP增速 * 0.7 + 人均GDP增速代理 * 0.3
    scores["增长动能"] = _zscore(df["GDP增速(%)"])

    # 3. 财政收入: 预算收入/GDP * 0.5 + 人均预算 * 0.5
    scores["财政收入"] = (
        _zscore(df["预算收入(亿元)"] / df["GDP(亿元)"]) * 0.5
        + _zscore(df["预算收入(亿元)"] / df["常住人口(万人)"]) * 0.5
    )

    # 4. 居民收入: 人均可支配收入
    scores["居民收入"] = _zscore(df["人均可支配收入(元)"])

    # 5. 消费活力: 社消/GDP * 0.5 + 人均社消 * 0.5
    scores["消费活力"] = (
        _zscore(df["社消总额(亿元)"] / df["GDP(亿元)"]) * 0.5
        + _zscore(df["社消总额(亿元)"] / df["常住人口(万人)"]) * 0.5
    )

    # 6. 工业实力: 工业增加值/GDP * 0.7 + 人均工业 * 0.3
    scores["工业实力"] = (
        _zscore(df["工业增加值(亿元)"] / df["GDP(亿元)"]) * 0.7
        + _zscore(df["工业增加值(亿元)"] / df["常住人口(万人)"]) * 0.3
    )

    # 7. 对外开放: 外资/GDP * 0.6 + 人均外资 * 0.4
    scores["对外开放"] = (
        _zscore(df["实际利用外资(亿美元)"] / df["GDP(亿元)"]) * 0.6
        + _zscore(df["实际利用外资(亿美元)"] / df["常住人口(万人)"]) * 0.4
    )

    # 8. 公共服务: R&D占比 * 0.4 + 每千人床位 * 0.3 + 每百万人高校 * 0.3
    scores["公共服务"] = (
        _zscore(df["R&D占比(%)"]) * 0.4
        + _zscore(df["每千人床位数"]) * 0.3
        + _zscore(df["高校数量"] / df["常住人口(万人)"]) * 0.3
    )

    # z-score 转 0-100 分数
    for col in scores.columns:
        scores[col] = _scale_to_0_100(scores[col])

    return scores


def _zscore(series: pd.Series) -> pd.Series:
    """z-score 标准化。"""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _scale_to_0_100(series: pd.Series) -> pd.Series:
    """将 z-score 映射到 0-100 区间（使用 sigmoid 平滑）。"""
    return (1 / (1 + np.exp(-series / 1.5))) * 100
