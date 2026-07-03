"""
Demo: 一线城市 + 新一线城市经济对比

用法:
    python examples/demo.py
"""

import sys
from pathlib import Path

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from city_compare.fetcher import compare_cities, compute_composite_scores
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
import pandas as pd


def main():
    print("=" * 60)
    print("  城市经济多维对比诊断 -- Demo")
    print("=" * 60)

    # ---- 第一部分: 一线城市 ----
    print("\n[1/5] 获取一线城市数据...")
    tier1 = ["上海", "北京", "深圳", "广州"]
    result_tier1 = compare_cities(tier1)
    print(f"    已获取 {len(tier1)} 个城市数据: {tier1}")

    print("\n[2/5] 计算综合得分...")
    scores_tier1 = compute_composite_scores(tier1)
    print("\n    一线城市综合得分:")
    for city in tier1:
        avg = scores_tier1.loc[city].mean()
        print(f"      {city}: {avg:.1f}")

    print("\n[3/5] 生成可视化...")
    # 雷达图
    city_score_dict = {city: scores_tier1.loc[city] for city in tier1}
    radar_fig = build_radar(city_score_dict, list(DIMENSION_LABELS.keys()))
    radar_path = Path(__file__).parent / "demo_radar.html"
    radar_fig.write_html(str(radar_path))
    print(f"    雷达图已保存: {radar_path}")

    # 排名图
    ranking_df = scores_tier1.mean(axis=1).reset_index()
    ranking_df.columns = ["城市", "综合得分"]
    ranking_df = ranking_df.sort_values("综合得分", ascending=False)
    bar_fig = build_bar_race(ranking_df, "综合得分")
    bar_path = Path(__file__).parent / "demo_bar.html"
    bar_fig.write_html(str(bar_path))
    print(f"    排名图已保存: {bar_path}")

    # 相关性热力图
    raw = result_tier1["raw"].set_index("城市")
    corr_fig = build_correlation_matrix(raw)
    corr_path = Path(__file__).parent / "demo_corr.html"
    corr_fig.write_html(str(corr_path))
    print(f"    热力图已保存: {corr_path}")

    # 聚类
    cluster_fig = build_city_clustering(raw, n_clusters=2)
    cluster_path = Path(__file__).parent / "demo_cluster.html"
    cluster_fig.write_html(str(cluster_path))
    print(f"    聚类图已保存: {cluster_path}")

    # 差距分析: 上海 vs 北京
    gap_fig = build_gap_analysis("上海", "北京", scores_tier1)
    gap_path = Path(__file__).parent / "demo_gap.html"
    gap_fig.write_html(str(gap_path))
    print(f"    差距图已保存: {gap_path}")

    # ---- 第二部分: 完整报告 ----
    print("\n[4/5] 生成综合报告...")
    report_path = Path(__file__).parent / "demo_report.html"
    generate_report(tier1, output_path=str(report_path))
    print(f"    报告已保存: {report_path}")

    # ---- 第三部分: 新一线对比 ----
    print("\n[5/5] 新一线城市快照...")
    new_tier1 = ["成都", "杭州", "武汉", "南京", "苏州", "重庆"]
    result_nt1 = compare_cities(new_tier1)
    scores_nt1 = compute_composite_scores(new_tier1)
    print(f"\n    新一线城市综合得分 Top 3:")
    avg_scores = scores_nt1.mean(axis=1).sort_values(ascending=False)
    for city in avg_scores.index[:3]:
        print(f"      {city}: {avg_scores[city]:.1f}")

    print("\n" + "=" * 60)
    print(f"  Demo 完成! 所有输出文件在: {Path(__file__).parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
