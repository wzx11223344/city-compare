"""
中国城市经济多维对比仪表板 -- Streamlit Web App

用法:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

# 页面配置 (必须在最前)
st.set_page_config(
    page_title="城市经济对比诊断",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from city_compare.fetcher import (
    compare_cities,
    compute_composite_scores,
    list_available_cities,
)
from city_compare.radar import (
    build_radar,
    build_bar_race,
    build_correlation_matrix,
    build_city_clustering,
    build_gap_analysis,
    build_rank_table,
    DIMENSION_LABELS,
)
from city_compare.report import generate_report

# ---- 自定义 CSS ----
st.markdown(
    """
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "PingFang SC", "SimHei", sans-serif;
    }
    /* 标题样式 */
    .main-title {
        font-size: 2.4em;
        font-weight: 800;
        background: linear-gradient(135deg, #1a2a6c, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .subtitle {
        color: #7f8c8d;
        font-size: 1.05em;
        margin-bottom: 20px;
    }
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa, #e8ecf1);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card .value {
        font-size: 1.6em;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-card .label {
        font-size: 0.8em;
        color: #7f8c8d;
        margin-top: 2px;
    }
    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("### 🏙️ 城市选择")
    all_cities = list_available_cities()

    # 快速选择预设
    preset = st.radio(
        "快速选择",
        ["自定义", "一线城市", "新一线 Top10", "长三角", "珠三角"],
        index=0,
        label_visibility="collapsed",
    )

    if preset == "自定义":
        default_cities = ["北京", "上海", "深圳", "广州"]
    elif preset == "一线城市":
        default_cities = ["北京", "上海", "深圳", "广州"]
    elif preset == "新一线 Top10":
        default_cities = [
            "成都", "杭州", "武汉", "南京", "重庆",
            "苏州", "西安", "长沙", "天津", "青岛",
        ]
    elif preset == "长三角":
        default_cities = [
            "上海", "杭州", "南京", "苏州", "宁波",
            "无锡", "合肥", "南通", "常州", "绍兴",
        ]
    elif preset == "珠三角":
        default_cities = ["深圳", "广州", "东莞", "佛山", "珠海", "惠州"]

    selected_cities = st.multiselect(
        "选择需要对比的城市 (2-12个)",
        options=all_cities,
        default=default_cities,
        max_selections=12,
    )

    year = st.selectbox("数据年份", [2023, 2022, 2021], index=0)

    st.markdown("---")
    st.markdown("### ⚙️ 设置")
    dark_mode = st.toggle("深色主题", value=False)

    st.markdown("---")
    st.markdown(
        """
    <div style="font-size:0.8em; color:#7f8c8d;">
    <b>city-compare v2.0</b><br>
    数据来源: 国家统计局, akshare<br>
    方法论: z-score 标准化 + 多指标复合
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---- 主页面 ----
st.markdown('<p class="main-title">中国城市经济多维对比诊断</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">结构化经济健康诊断工具 -- 雷达图 | 排名 | 聚类 | 差距分析 | 数据附录</p>',
    unsafe_allow_html=True,
)

if len(selected_cities) < 2:
    st.warning("请从侧边栏至少选择 2 个城市进行对比。")
    st.stop()

# ---- 加载数据 ----
@st.cache_data(ttl=600, show_spinner="正在获取数据...")
def load_data(cities_tuple, y):
    return compare_cities(list(cities_tuple), y)


@st.cache_data(ttl=600, show_spinner="正在计算综合得分...")
def load_scores(cities_tuple):
    return compute_composite_scores(list(cities_tuple))


cities_tuple = tuple(selected_cities)
with st.spinner(f"正在获取 {len(selected_cities)} 个城市的经济数据..."):
    result = load_data(cities_tuple, year)
    scores = load_scores(cities_tuple)

dimensions = list(DIMENSION_LABELS.keys())

# ---- 指标概览卡片 ----
st.markdown("### 📊 数据快照")
cols = st.columns(min(len(selected_cities), 8))
for i, city in enumerate(selected_cities):
    cfg_data = result["overview"]
    row = cfg_data[cfg_data["城市"] == city]
    if not row.empty:
        r = row.iloc[0]
        with cols[i % len(cols)]:
            gdp_str = f"{r['GDP(亿元)']:.0f}" if r["GDP(亿元)"] >= 10000 else f"{r['GDP(亿元)']:.1f}"
            growth_str = f"{r['GDP增速(%)']}%"
            st.markdown(
                f"""<div class="metric-card">
                <div style="font-weight:700;font-size:1em;margin-bottom:4px;color:#2c3e50;">{city}</div>
                <div class="value">{gdp_str}</div>
                <div class="label">GDP (亿元) | 增速 {growth_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ---- Tab 页 ----
tab_radar, tab_ranking, tab_cluster, tab_gap, tab_corr, tab_data = st.tabs(
    ["雷达图", "排名", "聚类分析", "差距对比", "相关性", "原始数据"]
)

with tab_radar:
    st.markdown("#### 8维综合雷达图")
    st.caption("每个维度由2-3个子指标加权合成，z-score标准化到0-100区间。面积越大 = 综合实力越强。")

    city_score_dict = {city: scores.loc[city] for city in selected_cities}
    radar_fig = build_radar(city_score_dict, dimensions, height=650)
    st.plotly_chart(radar_fig, use_container_width=True)

    # 综合得分排名
    st.markdown("#### 各维度得分明细")
    display_scores = scores.copy()
    display_scores.index.name = "城市"
    st.dataframe(
        display_scores.style.format("{:.1f}")
        .background_gradient(cmap="Blues", axis=1)
        .highlight_max(axis=0, color="#e8f5e9"),
        use_container_width=True,
        height=35 * (len(selected_cities) + 1) + 3,
    )

with tab_ranking:
    st.markdown("#### 综合得分排名")

    ranking_df = scores.mean(axis=1).reset_index()
    ranking_df.columns = ["城市", "综合得分"]
    ranking_df = ranking_df.sort_values("综合得分", ascending=False)

    bar_fig = build_bar_race(ranking_df, "综合得分", height=500)
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")
    rank_html = build_rank_table(ranking_df, "城市综合排名")
    st.components.v1.html(
        f"""
        <style>
        .rank-table-wrapper {{ font-family: "Microsoft YaHei", sans-serif; }}
        .rank-table {{ width:100%; border-collapse:collapse; font-size:0.95em; }}
        .rank-table thead th {{ background:#2c3e50; color:white; padding:12px 16px; text-align:center; }}
        .rank-table tbody td {{ padding:10px 16px; text-align:center; border-bottom:1px solid #ecf0f1; }}
        .rank-cell {{ font-weight:700; color:#3498db; }}
        .table-title {{ text-align:center; color:#2c3e50; }}
        </style>
        {rank_html}
        """,
        height=50 * (len(ranking_df) + 2) + 10,
        scrolling=True,
    )

with tab_cluster:
    st.markdown("#### 城市经济结构聚类 (PCA降维)")
    st.caption("K-means 聚类 + PCA 降维，将经济结构相似的城市归为一组。")

    raw = result["raw"].set_index("城市")
    n_clusters = st.slider("聚类数量", 2, min(5, len(selected_cities)), 3, key="cluster_n")
    cluster_fig = build_city_clustering(raw, n_clusters=n_clusters)
    st.plotly_chart(cluster_fig, use_container_width=True)

with tab_gap:
    st.markdown("#### 双城市差距对比")

    if len(selected_cities) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            city_a = st.selectbox("城市 A", selected_cities, index=0, key="gap_a")
        with col_b:
            city_b = st.selectbox("城市 B", selected_cities, index=1, key="gap_b")

        if city_a != city_b:
            gap_fig = build_gap_analysis(city_a, city_b, scores)
            st.plotly_chart(gap_fig, use_container_width=True)
        else:
            st.info("请选择不同的城市进行对比。")
    else:
        st.info("至少需要 2 个城市。")

with tab_corr:
    st.markdown("#### 城市经济指标相关性热力图")
    st.caption("Pearson 相关系数: 红色=正相关, 蓝色=负相关。")

    corr_fig = build_correlation_matrix(raw, height=650)
    st.plotly_chart(corr_fig, use_container_width=True)

with tab_data:
    st.markdown("#### 原始数据附录")

    for key in ["overview", "per_capita", "structure", "public_service", "raw"]:
        df = result[key]
        st.markdown(f"**{key.upper()}**")
        st.dataframe(
            df.style.format("{:.2f}", na_rep="-"),
            use_container_width=True,
            height=35 * (len(df) + 1) + 3,
        )
        st.markdown("---")

# ---- 下载报告 ----
st.markdown("---")
st.markdown("### 📥 导出完整报告")

col_dl, col_info = st.columns([1, 3])
with col_dl:
    if st.button("生成 HTML 诊断报告", type="primary", use_container_width=True):
        with st.spinner("正在生成报告..."):
            html_content = generate_report(selected_cities, year=year)
            st.download_button(
                label="下载 HTML 报告",
                data=html_content,
                file_name=f"city_compare_{'_'.join(selected_cities[:3])}_{year}.html",
                mime="text/html",
                use_container_width=True,
            )
with col_info:
    st.caption(
        "报告包含: 封面 → 雷达图 → 排名表 → 聚类图 → 相关性矩阵 → 数据附录。"
        "自包含 HTML，无需额外依赖即可在浏览器中查看。"
    )
