# -*- coding: utf-8 -*-
"""
朗逸补货与促销优化（改进版）

相比原版的主要修正：
1. 利润只按【实际销量】计算，不再把"补进来的车"全部算成利润；
   未售出的库存按持有/处置成本扣减，避免模型"无脑多补货到上限"。
2. 需求预测从数据中读取，提供三种口径：
   - seasonal_naive：去年同期（季节性朴素）
   - last3_mean    ：近 3 个月均值
   - last3_actual  ：近 3 个月实际销量平移（原版做法）
3. 自动输出"不促销"基线方案做对比，并可选做促销力度的敏感性分析，
   验证"促销到底值不值"。
4. 代码结构化：配置集中、函数拆分、结果可导出 CSV。

运行前提（pip 安装缺失的库即可）：
    pip install pandas openpyxl pulp
"""

import os
import sys
import pulp
import pandas as pd

# ---------------- 配置 ----------------
CONFIG = {
    "data_path": r"C:\Users\崔\Desktop\项目二\中国汽车分车型每月销售量.xlsx",
    "model_name": "朗逸",
    "n_months": 3,                        # 规划未来几个月
    "profit_margin": 0.12,                # 单车利润率假设
    "promo_profit_factor": 0.85,          # 促销时单车利润 = 正常利润 × 0.85
    "promo_demand_lift": 1.25,            # 促销需求增幅（1.25 = +25%）
    "inventory_ratio": 1.2,               # 总补货上限 = 预测总需求 × 1.2
    "holding_cost": 200.0,                # 每辆未售出库存的持有/处置成本（元）
    "order_limit": None,                  # 单月补货上限（辆），None 表示不限制
    "forecast_method": "seasonal_naive",  # seasonal_naive / last3_mean / last3_actual
    "scenario": 1.0,                      # 需求情景系数：0.9 悲观 / 1.0 中性 / 1.1 乐观
    "sensitivity_promo": True,            # 是否做促销力度敏感性分析
    "out_csv": os.path.join(os.path.dirname(os.path.abspath(__file__)), "朗逸补货方案.csv"),
}


# ---------------- 数据读取 ----------------
def load_data(path):
    """读取销量表并做基本清洗（去掉重复表头、转数值）。"""
    df = pd.read_excel(path)
    df = df[df["年份"] != "年份"].copy()          # 原表第一行数据是重复表头
    df["销量"] = pd.to_numeric(df["销量"], errors="coerce")
    df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
    df["月份"] = pd.to_numeric(df["月份"], errors="coerce")
    return df.dropna(subset=["销量", "年份", "月份"])


def price_mid(s):
    """把"9.40-15.19"这类区间价取中值。"""
    if isinstance(s, str) and "-" in s:
        lo, hi = s.split("-")
        try:
            return (float(lo) + float(hi)) / 2
        except ValueError:
            return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def future_months(hist, n):
    """返回未来 n 个月的 (年份, 月份) 列表。"""
    last_year, last_month = int(hist["年份"].iloc[-1]), int(hist["月份"].iloc[-1])
    base = last_year * 12 + (last_month - 1)   # 最后一条数据所在的"月序号"（2023-09 = 24284）
    out = []
    for k in range(1, n + 1):
        idx = base + k                          # 下一个月、再下一个月……
        out.append((idx // 12, idx % 12 + 1))
    return out


def forecast_sales(hist, method, n=3):
    """按所选方法预测未来 n 个月的需求（辆）。"""
    hist = hist.sort_values(["年份", "月份"])
    pred = []
    for k in range(1, n + 1):
        y, m = future_months(hist, n)[k - 1]
        same = hist[(hist["年份"] == y - 1) & (hist["月份"] == m)]
        same_v = float(same["销量"].iloc[0]) if len(same) else None
        mean_v = float(hist["销量"].tail(3).mean())
        if method == "seasonal_naive":
            v = same_v if same_v is not None else mean_v
        elif method == "last3_mean":
            v = mean_v
        elif method == "last3_actual":
            v = float(hist["销量"].iloc[-k])   # 原版做法：近3月实际平移
        else:
            raise ValueError(f"未知预测方法: {method}")
        pred.append(v)
    return pred


# ---------------- 优化模型 ----------------
def build_model(pred, normal_profit, promo_profit, cfg):
    """
    决策变量：
      x[i] 第 i 个月补货量
      z[i] 第 i 个月是否促销（0/1）
      s[i] 第 i 个月实际销量（<= 补货量，<= 促销后的需求）
      w[i] 促销价卖出的数量（线性化 w = z * s）
    目标：正常利润×销量 - 促销让利×促销价销量 - 持有成本×未售出
    约束：总补货量 <= 库存上限；单月补货上限（可选）。
    """
    T = range(len(pred))
    cap = int(sum(pred) * cfg["inventory_ratio"])
    M = cap + 1                              # 线性化用的大 M
    lift = cfg["promo_demand_lift"]

    prob = pulp.LpProblem("朗逸补货与促销优化", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("补货量", T, lowBound=0, cat="Integer")
    z = pulp.LpVariable.dicts("是否促销", T, cat="Binary")
    s = pulp.LpVariable.dicts("销量", T, lowBound=0, cat="Integer")
    w = pulp.LpVariable.dicts("促销价销量", T, lowBound=0, cat="Integer")

    prob += (
        pulp.lpSum(normal_profit * s[i] for i in T)
        - pulp.lpSum((normal_profit - promo_profit) * w[i] for i in T)
        - pulp.lpSum(cfg["holding_cost"] * (x[i] - s[i]) for i in T)
    )

    for i in T:
        # 销量不能超过补货量（简化：当月补当月卖，不跨月结转）
        prob += s[i] <= x[i]
        # 销量不能超过促销后的需求：d = pred * (1 + (lift-1) * z)
        prob += s[i] <= pred[i] + pred[i] * (lift - 1) * z[i]
        # 线性化 w = z * s
        prob += w[i] <= s[i]
        prob += w[i] <= M * z[i]
        prob += w[i] >= s[i] - M * (1 - z[i])
        if cfg["order_limit"]:
            prob += x[i] <= cfg["order_limit"]

    prob += pulp.lpSum(x[i] for i in T) <= cap
    return prob, x, z, s, w, cap


def solve_once(pred, normal_profit, promo_profit, cfg):
    prob, x, z, s, w, cap = build_model(pred, normal_profit, promo_profit, cfg)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    rows = []
    for i in range(len(pred)):
        is_promo = pulp.value(z[i]) > 0.5
        sold = int(round(pulp.value(s[i])))
        rows.append({
            "预测需求": int(round(pred[i])),
            "是否促销": int(is_promo),
            "补货量": int(round(pulp.value(x[i]))),
            "预计销量": sold,
            "单车利润": float(promo_profit if is_promo else normal_profit),
            "本月利润": sold * (promo_profit if is_promo else normal_profit),
        })
    rows.append({
        "预测需求": int(round(sum(pred))),
        "是否促销": None,
        "补货量": int(round(pulp.value(pulp.lpSum(x[i] for i in range(len(pred)))))),
        "预计销量": int(round(pulp.value(pulp.lpSum(s[i] for i in range(len(pred)))))),
        "单车利润": None,
    })
    return {"rows": rows, "profit": pulp.value(prob.objective)}


def baseline_profit(pred, normal_profit, cap):
    """不促销基线的最大利润：按预测需求全部满足、按需补货。"""
    total_demand = sum(pred)
    sold = min(total_demand, cap)
    return normal_profit * sold


# ---------------- 主流程 ----------------
def main():
    cfg = CONFIG
    df = load_data(cfg["data_path"])
    hist = df[df["车型"] == cfg["model_name"]].sort_values(["年份", "月份"]).copy()
    if hist.empty:
        sys.exit(f"数据中找不到车型：{cfg['model_name']}")

    # 单车利润
    hist["售价中值_万元"] = hist["售价（万元）"].astype(str).apply(price_mid)
    avg_price = hist["售价中值_万元"].mean()
    normal_profit = round(avg_price * 10000 * cfg["profit_margin"])   # 取整到元，避免 12,540.9 的显示误差
    promo_profit = round(normal_profit * cfg["promo_profit_factor"])

    # 预测
    pred = forecast_sales(hist, cfg["forecast_method"], cfg["n_months"])
    pred = [v * cfg["scenario"] for v in pred]
    months = future_months(hist, cfg["n_months"])
    last_year, last_month = int(hist["年份"].iloc[-1]), int(hist["月份"].iloc[-1])
    if (months[0][0], months[0][1]) <= (last_year, last_month):
        sys.exit(f"预测月份计算异常：{months[0]} 应晚于数据截止月份 {last_year}-{last_month}。")
    cap = int(sum(pred) * cfg["inventory_ratio"])

    print(f"车型：{cfg['model_name']}  数据截止：{int(hist['年份'].iloc[-1])}-{int(hist['月份'].iloc[-1])}")
    print(f"平均售价中值：{avg_price:.2f} 万元   正常单车利润：{normal_profit:.0f} 元   促销单车利润：{promo_profit:.0f} 元\n")
    print(f"预测方法：{cfg['forecast_method']}（情景系数 {cfg['scenario']}）")
    print(f"未来 {cfg['n_months']} 个月预测需求：{[f'{m[0]}-{m[1]:02d}: {int(v):,}' for m, v in zip(months, pred)]}\n")

    res = solve_once(pred, normal_profit, promo_profit, cfg)
    if res is None:
        sys.exit("模型求解失败，请检查参数。")
    base = baseline_profit(pred, normal_profit, cap)

    print(f"总补货上限：{cap:,} 辆")
    print(f"不促销基线利润：{base:,.0f} 元")
    print(f"优化方案利润：{res['profit']:,.0f} 元（相对基线 {res['profit'] / base - 1:+.2%}）\n")
    print("逐月方案：")
    print(f"  {'月份':<10}{'预测需求':>10}{'促销':>6}{'补货量':>10}{'预计销量':>10}{'单车利润':>10}{'本月利润':>14}")
    for i, (y, m) in enumerate(months):
        r = res["rows"][i]
        print(f"  {f'{y}-{m:02d}':<10}{r['预测需求']:>10,}{'是' if r['是否促销'] else '否':>6}{r['补货量']:>10,}{r['预计销量']:>10,}{r['单车利润']:>10,.0f}{r['本月利润']:>14,}")
    t = res["rows"][-1]
    month_profit = sum(r["本月利润"] for r in res["rows"][:-1])
    print(f"  {'合计':<10}{t['预测需求']:>10,}{'-':>6}{t['补货量']:>10,}{t['预计销量']:>10,}{'-':>10}{month_profit:>14,}")

    # 导出 CSV
    rows_out = []
    for i, (y, m) in enumerate(months):
        r = res["rows"][i]
        rows_out.append({
            "月份": f"{y}-{m:02d}",
            "预测需求": r["预测需求"],
            "是否促销": "是" if r["是否促销"] else "否",
            "补货量": r["补货量"],
            "预计销量": r["预计销量"],
            "单车利润": r["单车利润"],
            "本月利润": r["本月利润"],
        })
    rows_out.append({"月份": "合计", "预测需求": t["预测需求"], "是否促销": "",
                     "补货量": t["补货量"], "预计销量": t["预计销量"],
                     "单车利润": "", "本月利润": month_profit})
    pd.DataFrame(rows_out).to_csv(cfg["out_csv"], index=False, encoding="utf-8-sig")
    print(f"\n方案已导出：{cfg['out_csv']}")

    # 促销力度敏感性
    if cfg["sensitivity_promo"]:
        print("\n=== 促销力度敏感性（需求增幅 × 单车利润让利 是否划算）===")
        print(f"  盈亏平衡增幅 = 正常利润/促销利润 - 1 = {normal_profit / promo_profit - 1:.1%}")
        print(f"  {'需求增幅':>8}{'最优促销月份':>14}{'优化利润':>18}{'相对基线':>10}")
        for lift in (1.10, 1.15, 1.18, 1.20, 1.25, 1.30, 1.40):
            cfg2 = dict(cfg, promo_demand_lift=lift)
            r2 = solve_once(pred, normal_profit, promo_profit, cfg2)
            if r2 is None:
                continue
            promo_months = ",".join(
                f"{months[i][0]}-{months[i][1]:02d}"
                for i in range(len(pred)) if r2["rows"][i]["是否促销"]
            ) or "无"
            print(f"  {lift - 1:>+8.0%}{promo_months:>16}{r2['profit']:>18,.0f}{r2['profit'] / base - 1:>+10.2%}")


if __name__ == "__main__":
    main()
