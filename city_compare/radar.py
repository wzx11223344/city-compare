"""
可视化核心模块

提供多城市雷达图、排名条形图、相关性热力图、聚类散点图、差距分析和排名表。
所有图表基于 Plotly，支持交互式悬停提示、缩放和 PNG 导出。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- 常量 ----

DIMENSION_LABELS = {
    "经济规模": "经济规模",
    "增长动能": "增长动能",
    "财政收入": "财政收入",
    "居民收入": "居民收入",
    "消费活力": "消费活力",
    "工业实力": "工业实力",
    "对外开放": "对外开放",
    "公共服务": "公共服务",
}

# 12 色调色板 (城市颜色映射)
COLORS = [
    "#E74C3C",  # 红
    "#3498DB",  # 蓝
    "#2ECC71",  # 绿
    "#F39C12",  # 橙
    "#9B59B6",  # 紫
    "#1ABC9C",  # 青
    "#E67E22",  # 深橙
    "#2980B9",  # 深蓝
    "#27AE60",  # 深绿
    "#C0392B",  # 暗红
    "#8E44AD",  # 暗紫
    "#16A085",  # 暗青
]

_CITY_COLOR_MAP: Dict[str, str] = {}


def _get_city_colors(cities: List[str]) -> Dict[str, str]:
    """为每个城市分配唯一颜色。"""
    global _CITY_COLOR_MAP
    for i, city in enumerate(cities):
        if city not in _CITY_COLOR_MAP:
            _CITY_COLOR_MAP[city] = COLORS[i % len(COLORS)]
    return {c: _CITY_COLOR_MAP.get(c, COLORS[0]) for c in cities}


# ---- 雷达图 ----

def build_radar(
    city_data_dict: Dict[str, pd.Series],
    dimensions: Optional[List[str]] = None,
    title: str = "城市经济多维雷达对比",
    height: int = 700,
) -> go.Figure:
    """
    构建多城市雷达对比图。

    Args:
        city_data_dict: {城市名: pd.Series(维度值)} 如 scores DataFrame 的行
        dimensions: 维度列表（默认使用全部8个维度）
        title: 图表标题
        height: 图表高度（像素）

    Returns:
        Plotly Figure 对象
    """
    if dimensions is None:
        dimensions = list(DIMENSION_LABELS.keys())

    colors = _get_city_colors(list(city_data_dict.keys()))
    fig = go.Figure()
    dim_labels = [DIMENSION_LABELS.get(d, d) for d in dimensions]

    for city, scores in city_data_dict.items():
        values = [scores.get(d, 0) for d in dimensions]
        # 闭合雷达图
        values_closed = values + [values[0]]
        dims_closed = dim_labels + [dim_labels[0]]

        color = colors[city]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=dims_closed,
                name=city,
                fill="toself",
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, symbol="circle"),
                hovertemplate=(
                    f"<b>{city}</b><br>"
                    "%{theta}: <b>%{r:.1f}</b><br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            x=0.5,
            xanchor="center",
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linewidth=1,
                linecolor="#BDC3C7",
                gridcolor="#ECF0F1",
                tickfont=dict(size=10, color="#7F8C8D"),
                ticks="outside",
                ticklen=4,
            ),
            angularaxis=dict(
                tickfont=dict(size=13, family="Microsoft YaHei, SimHei, sans-serif", color="#34495E"),
                linewidth=1,
                linecolor="#BDC3C7",
                gridcolor="#ECF0F1",
            ),
            bgcolor="rgba(255,255,255,0.95)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family="Microsoft YaHei, SimHei, sans-serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E0E0E0",
            borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=80, r=80, t=80, b=80),
    )

    return fig


# ---- 排名条形图 ----

def build_bar_race(
    city_ranking: pd.DataFrame,
    dimension: str,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    构建横向排名条形图。

    Args:
        city_ranking: 包含 '城市' 列和维度值列的 DataFrame
        dimension: 用于排名的列名
        title: 图表标题
        height: 图表高度
    """
    if title is None:
        title = f"{DIMENSION_LABELS.get(dimension, dimension)} 城市排名"

    df = city_ranking.sort_values(dimension, ascending=True)
    cities = df["城市"].tolist()
    values = df[dimension].tolist()
    colors = _get_city_colors(cities)

    bar_colors = [colors[c] for c in cities]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=cities,
            x=values,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="white", width=1),
                opacity=0.9,
            ),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=13, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="得分", font=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif")),
            showgrid=True,
            gridcolor="#ECF0F1",
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",
            tickfont=dict(size=13, family="Microsoft YaHei, SimHei, sans-serif"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=80, r=80, t=60, b=40),
        showlegend=False,
    )

    return fig


# ---- 相关性热力图 ----

def build_correlation_matrix(
    city_stats: pd.DataFrame,
    title: str = "城市经济指标相关性热力图",
    height: int = 600,
) -> go.Figure:
    """
    构建指标间相关性热力图。

    Args:
        city_stats: 包含数值指标的 DataFrame (index=城市)
        title: 标题
        height: 高度
    """
    corr = city_stats.corr()

    # 中文标签映射
    label_map = {
        "GDP(亿元)": "GDP规模",
        "GDP增速(%)": "GDP增速",
        "人均GDP(元)": "人均GDP",
        "常住人口(万人)": "人口",
        "预算收入(亿元)": "预算收入",
        "人均可支配收入(元)": "人均收入",
        "社消总额(亿元)": "消费总额",
        "工业增加值(亿元)": "工业增加值",
        "实际利用外资(亿美元)": "利用外资",
        "高校数量": "高校数量",
        "R&D占比(%)": "R&D占比",
        "每千人床位数": "千人床位",
    }
    corr.index = [label_map.get(i, i) for i in corr.index]
    corr.columns = [label_map.get(c, c) for c in corr.columns]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, "#2C3E50"],
                [0.25, "#3498DB"],
                [0.5, "#ECF0F1"],
                [0.75, "#E74C3C"],
                [1.0, "#C0392B"],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            hoverongaps=False,
            hovertemplate="%{x} vs %{y}: <b>%{z:.3f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(tickfont=dict(size=11, family="Microsoft YaHei, SimHei, sans-serif"), side="bottom"),
        yaxis=dict(tickfont=dict(size=11, family="Microsoft YaHei, SimHei, sans-serif")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=100, r=60, t=60, b=100),
    )

    return fig


# ---- K-means 聚类散点图 ----

def build_city_clustering(
    city_stats: pd.DataFrame,
    n_clusters: int = 3,
    title: str = "城市经济发展类型聚类 (PCA降维)",
    height: int = 600,
) -> go.Figure:
    """
    K-means 聚类 + PCA 降维可视化。

    Args:
        city_stats: 数值指标 DataFrame
        n_clusters: 聚类数
        title: 标题
        height: 高度
    """
    if len(city_stats) < n_clusters:
        n_clusters = max(1, len(city_stats) - 1)

    numeric = city_stats.select_dtypes(include=[np.number])

    # 标准化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)

    colors = _get_city_colors(city_stats.index.tolist())

    fig = go.Figure()

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_cities = city_stats.index[mask].tolist()
        xs = coords[mask, 0]
        ys = coords[mask, 1]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=f"聚类 {cluster_id + 1}",
                text=cluster_cities,
                textposition="top center",
                textfont=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
                marker=dict(
                    size=18,
                    color=COLORS[cluster_id % len(COLORS)],
                    opacity=0.8,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="<b>%{text}</b><br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}<br>"
                "聚类: %{name}<extra></extra>",
            )
        )

    var_exp = pca.explained_variance_ratio_

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text=f"主成分 1 ({var_exp[0]*100:.1f}%)", font=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif")),
            zeroline=True,
            zerolinecolor="#BDC3C7",
        ),
        yaxis=dict(
            title=dict(text=f"主成分 2 ({var_exp[1]*100:.1f}%)", font=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif")),
            zeroline=True,
            zerolinecolor="#BDC3C7",
        ),
        legend=dict(
            font=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


# ---- 差距分析 ----

def build_gap_analysis(
    city_a: str,
    city_b: str,
    scores: pd.DataFrame,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    双城市差距对比图（双向条形图）。

    Args:
        city_a: 城市 A
        city_b: 城市 B
        scores: 综合得分 DataFrame (index=城市, columns=维度)
        title: 标题
        height: 高度
    """
    if title is None:
        title = f"{city_a} vs {city_b} 差距分析"

    dims = list(DIMENSION_LABELS.keys())
    dim_labels = [DIMENSION_LABELS[d] for d in dims]

    vals_a = [scores.loc[city_a, d] if d in scores.columns else 0 for d in dims]
    vals_b = [scores.loc[city_b, d] if d in scores.columns else 0 for d in dims]
    diffs = [a - b for a, b in zip(vals_a, vals_b)]

    color_a = _get_city_colors([city_a])[city_a]
    color_b = _get_city_colors([city_b])[city_b]

    fig = go.Figure()

    # 城市 A 正值条
    fig.add_trace(
        go.Bar(
            y=dim_labels,
            x=[max(0, d) for d in diffs],
            orientation="h",
            name=f"{city_a} 领先",
            marker=dict(color=color_a, opacity=0.85),
            hovertemplate=f"{city_a} 领先 %{{x:.1f}}<extra></extra>",
        )
    )

    # 城市 B 负值条
    fig.add_trace(
        go.Bar(
            y=dim_labels,
            x=[min(0, d) for d in diffs],
            orientation="h",
            name=f"{city_b} 领先",
            marker=dict(color=color_b, opacity=0.85),
            hovertemplate=f"{city_b} 领先 %{{x:.1f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Microsoft YaHei, SimHei, sans-serif", color="#2C3E50"),
            x=0.5,
        ),
        barmode="relative",
        xaxis=dict(
            title=dict(text="得分差距 (正数=A领先, 负数=B领先)", font=dict(size=11, family="Microsoft YaHei, SimHei, sans-serif")),
            zeroline=True,
            zerolinecolor="#34495E",
            zerolinewidth=2,
        ),
        yaxis=dict(
            tickfont=dict(size=13, family="Microsoft YaHei, SimHei, sans-serif"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Microsoft YaHei, SimHei, sans-serif"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=100, r=60, t=60, b=40),
    )

    return fig


# ---- 排名表 (HTML) ----

def build_rank_table(
    city_ranking: pd.DataFrame,
    title: str = "城市综合排名",
) -> str:
    """
    生成样式化的 HTML 排名表。

    Args:
        city_ranking: 包含 '城市' 和数值列的 DataFrame
        title: 表格标题

    Returns:
        HTML 字符串
    """
    df = city_ranking.copy()
    # 按第一列数值降序排列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df = df.sort_values(numeric_cols[0], ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "排名"

    colors = _get_city_colors(df["城市"].tolist())

    html = f"""
    <div class="rank-table-wrapper">
        <h3 class="table-title">{title}</h3>
        <table class="rank-table">
            <thead>
                <tr>
                    <th>排名</th>
    """
    for col in df.columns:
        html += f"<th>{col}</th>\n"
    html += "</tr></thead><tbody>\n"

    for rank, (idx, row) in enumerate(df.iterrows(), start=1):
        city = row.get("城市", "")
        city_color = colors.get(city, "#333")
        html += f"""<tr>
                    <td class="rank-cell">{rank}</td>
        """
        for col in df.columns:
            val = row[col]
            if col == "城市":
                html += f'<td class="city-cell" style="color:{city_color}; font-weight:700;">{val}</td>'
            elif isinstance(val, float):
                html += f"<td>{val:.1f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>\n"

    html += """
            </tbody>
        </table>
    </div>
    """
    return html
