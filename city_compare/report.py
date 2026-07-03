"""
自包含 HTML 报告生成器

从对比结果生成完整的单文件 HTML 报告。
包含: 标题页 -> 雷达图 -> 排名表 -> 聚类图 -> 相关性热力图 -> 数据附录
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from city_compare.fetcher import compare_cities, compute_composite_scores
from city_compare.radar import (
    build_radar,
    build_bar_race,
    build_correlation_matrix,
    build_city_clustering,
    build_rank_table,
    DIMENSION_LABELS,
)


def _fig_to_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    """将 Plotly Figure 转为 HTML 片段。"""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": 800,
                "width": 1200,
                "scale": 2,
            },
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def generate_report(
    city_list: List[str],
    output_path: Optional[str] = None,
    year: int = 2023,
    title: str = "中国城市经济多维对比诊断报告",
) -> str:
    """
    生成完整的自包含 HTML 对比报告。

    Args:
        city_list: 待对比的城市名称列表
        output_path: 输出文件路径（None 则返回 HTML 字符串）
        year: 数据年份
        title: 报告标题

    Returns:
        HTML 字符串；如果指定了 output_path 则同时写入文件。
    """
    # ---- 数据准备 ----
    result = compare_cities(city_list, year)
    scores = compute_composite_scores(city_list)
    dimensions = list(DIMENSION_LABELS.keys())

    # ---- 图表生成 ----
    # 雷达图
    city_score_dict = {city: scores.loc[city] for city in city_list}
    radar_fig = build_radar(city_score_dict, dimensions, f"{title} - 雷达对比")

    # 排名条形图 (综合得分)
    ranking_df = scores.mean(axis=1).reset_index()
    ranking_df.columns = ["城市", "综合得分"]
    ranking_df = ranking_df.sort_values("综合得分", ascending=False)
    bar_fig = build_bar_race(ranking_df, "综合得分", "城市综合得分排名")

    # 相关性热力图
    raw = result["raw"].set_index("城市")
    corr_fig = build_correlation_matrix(raw)

    # 聚类散点图
    cluster_fig = build_city_clustering(raw, n_clusters=min(3, len(city_list)))

    # 排名表 HTML
    rank_table_html = build_rank_table(ranking_df, "综合排名")

    # 数据附录表
    data_overview_html = _build_data_table(result["overview"], "综合概览")
    data_percapita_html = _build_data_table(result["per_capita"], "人均指标")
    data_structure_html = _build_data_table(result["structure"], "经济结构")

    # ---- 组装 HTML ----
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{plotly_cdn}
<style>
    :root {{
        --bg: #ffffff;
        --bg-alt: #f8f9fa;
        --text: #2c3e50;
        --text-muted: #7f8c8d;
        --border: #e0e0e0;
        --accent: #3498db;
        --accent2: #e74c3c;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "SimHei", sans-serif;
        background: var(--bg-alt);
        color: var(--text);
        line-height: 1.7;
        -webkit-font-smoothing: antialiased;
    }}

    /* ---- 封面 ---- */
    .cover {{
        background: linear-gradient(135deg, #1a2a6c 0%, #2c3e50 40%, #3498db 100%);
        color: white;
        padding: 80px 60px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    .cover::before {{
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 70%, rgba(255,255,255,0.05) 0%, transparent 50%);
    }}
    .cover h1 {{
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 16px;
        position: relative;
        letter-spacing: 2px;
    }}
    .cover .subtitle {{
        font-size: 1.3em;
        opacity: 0.9;
        margin-bottom: 30px;
        position: relative;
    }}
    .cover .cities {{
        display: flex;
        justify-content: center;
        gap: 16px;
        flex-wrap: wrap;
        position: relative;
        margin-bottom: 24px;
    }}
    .cover .city-tag {{
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 30px;
        padding: 8px 24px;
        font-size: 1.1em;
        font-weight: 600;
        backdrop-filter: blur(4px);
    }}
    .cover .meta {{
        font-size: 0.95em;
        opacity: 0.7;
        position: relative;
    }}

    /* ---- 章节 ----
    .section {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 50px 30px;
    }}
    .section-header {{
        text-align: center;
        margin-bottom: 36px;
    }}
    .section-header h2 {{
        font-size: 1.9em;
        font-weight: 700;
        color: #2c3e50;
        position: relative;
        display: inline-block;
    }}
    .section-header h2::after {{
        content: "";
        display: block;
        width: 60px;
        height: 4px;
        background: var(--accent);
        margin: 12px auto 0;
        border-radius: 2px;
    }}
    .section-header p {{
        color: var(--text-muted);
        margin-top: 8px;
        font-size: 1em;
    }}

    /* ---- 图表容器 ----
    .chart-container {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 30px;
        margin-bottom: 30px;
        overflow: hidden;
    }}
    .chart-container.full-width {{
        max-width: none;
    }}

    /* ---- 排名表 ----
    .rank-table-wrapper {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 30px;
        overflow-x: auto;
    }}
    .table-title {{
        font-size: 1.3em;
        margin-bottom: 20px;
        text-align: center;
        color: #2c3e50;
    }}
    .rank-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95em;
    }}
    .rank-table thead th {{
        background: #2c3e50;
        color: white;
        padding: 12px 16px;
        text-align: center;
        font-weight: 600;
        position: sticky;
        top: 0;
    }}
    .rank-table tbody td {{
        padding: 10px 16px;
        text-align: center;
        border-bottom: 1px solid #ecf0f1;
    }}
    .rank-table tbody tr:hover {{
        background: #f0f7ff;
    }}
    .rank-cell {{
        font-weight: 700;
        color: var(--accent);
    }}
    .city-cell {{
        font-weight: 700;
    }}

    /* ---- 双栏 ----
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
    }}
    @media (max-width: 900px) {{
        .two-col {{ grid-template-columns: 1fr; }}
        .cover h1 {{ font-size: 2em; }}
    }}

    /* ---- 附录 ----
    .appendix {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 50px 30px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85em;
        margin-bottom: 30px;
    }}
    .data-table thead th {{
        background: #34495e;
        color: white;
        padding: 10px 12px;
    }}
    .data-table tbody td {{
        padding: 8px 12px;
        border-bottom: 1px solid #ecf0f1;
        text-align: right;
    }}
    .data-table tbody td:first-child {{
        text-align: left;
        font-weight: 600;
    }}
    .data-table tbody tr:nth-child(even) {{
        background: #f8f9fa;
    }}

    /* ---- 打印 ----
    @media print {{
        body {{ background: white; }}
        .cover {{ page-break-after: always; }}
        .section {{ page-break-inside: avoid; padding: 20px; }}
        .chart-container {{ box-shadow: none; break-inside: avoid; }}
    }}

    /* ---- 页脚 ----
    .footer {{
        text-align: center;
        padding: 40px;
        color: var(--text-muted);
        font-size: 0.85em;
        border-top: 1px solid var(--border);
        max-width: 1200px;
        margin: 0 auto;
    }}
</style>
</head>
<body>

<!-- ====== 封面 ====== -->
<div class="cover">
    <h1>{title}</h1>
    <div class="subtitle">结构化多维经济健康诊断工具</div>
    <div class="cities">
        {''.join(f'<span class="city-tag">{c}</span>' for c in city_list)}
    </div>
    <div class="meta">
        数据年份: {year} &nbsp;|&nbsp; 指标维度: {len(dimensions)} &nbsp;|&nbsp;
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>

<!-- ====== 雷达图 ====== -->
<div class="section">
    <div class="section-header">
        <h2>多维度雷达对比</h2>
        <p>8个维度综合评估，z-score归一化至0-100区间</p>
    </div>
    <div class="chart-container">
        {_fig_to_html(radar_fig)}
    </div>
</div>

<!-- ====== 排名 ====== -->
<div class="section">
    <div class="section-header">
        <h2>综合排名</h2>
        <p>基于8个维度的加权均值得分排序</p>
    </div>
    <div class="two-col">
        <div class="chart-container">
            {_fig_to_html(bar_fig)}
        </div>
        {rank_table_html}
    </div>
</div>

<!-- ====== 聚类 ====== -->
<div class="section">
    <div class="section-header">
        <h2>城市发展类型聚类</h2>
        <p>基于K-means + PCA降维的经济结构相似性分组</p>
    </div>
    <div class="chart-container">
        {_fig_to_html(cluster_fig)}
    </div>
</div>

<!-- ====== 相关性 ====== -->
<div class="section">
    <div class="section-header">
        <h2>指标相关性矩阵</h2>
        <p>各经济指标间的Pearson相关系数</p>
    </div>
    <div class="chart-container">
        {_fig_to_html(corr_fig)}
    </div>
</div>

<!-- ====== 数据附录 ====== -->
<div class="appendix">
    <div class="section-header">
        <h2>数据附录</h2>
        <p>所有指标原始数据</p>
    </div>
    {data_overview_html}
    {data_percapita_html}
    {data_structure_html}
</div>

<div class="footer">
    <p>city-compare &mdash; 城市经济多维对比诊断工具</p>
    <p>数据来源: 中国国家统计局、各城市统计年鉴 &amp; akshare</p>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

</body>
</html>"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


def _build_data_table(df: pd.DataFrame, caption: str) -> str:
    """将 DataFrame 转为样式化 HTML 表格。"""
    html = f'<h3 style="margin-top:24px;color:#2c3e50;">{caption}</h3>\n'
    html += '<table class="data-table"><thead><tr>\n'
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>\n"
    for _, row in df.iterrows():
        html += "<tr>"
        for i, val in enumerate(row):
            if isinstance(val, float):
                html += f"<td>{val:.2f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>\n"
    html += "</tbody></table>\n"
    return html
