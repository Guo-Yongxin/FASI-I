# fasti_os_check.py
import json
import math
import numpy as np
import pandas as pd
import inspect


# =========================================
# 0) 路径
# =========================================
DEPLOY_DIR = "FASTI_OS_deploy"


# =========================================
# 1) 读取部署文件
# =========================================
coef_df = pd.read_csv(f"{DEPLOY_DIR}/coef_os.csv")
xcols_df = pd.read_csv(f"{DEPLOY_DIR}/x_columns_os.csv")
base_df = pd.read_csv(f"{DEPLOY_DIR}/baseline_survival_os.csv")
test_df = pd.read_csv(f"{DEPLOY_DIR}/test_cases_os.csv")

with open(f"{DEPLOY_DIR}/model_meta_os.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

beta_map = dict(zip(coef_df["term"], coef_df["beta"]))
x_col_order = xcols_df["x_col_order"].tolist()
center = float(meta["center"])
knots = meta["rcs"]["knots"]
center = float(meta["center"])  # [-0.088823529, -0.037810232, -0.008348214]

print("=== center used at runtime ===")
print(repr(center))
# =========================================
# 2) 线性插值 baseline survival
#    网页以后也用这个
# =========================================
def baseline_survival_at_time(t: float) -> float:
    times = base_df["time"].to_numpy(dtype=float)
    survs = base_df["baseline_survival"].to_numpy(dtype=float)

    if t <= times[0]:
        return float(survs[0])
    if t >= times[-1]:
        return float(survs[-1])

    return float(np.interp(t, times, survs))


# =========================================
# 3) RCS 基函数（与 Harrell restricted cubic spline 一致）
#    这里 k=3 knots -> 产生 2 个基函数项：
#    x 和 x'
# =========================================
def _pos_cube(z: float) -> float:
    return max(z, 0.0) ** 3


def rcs_basis_3knots(x: float, knots_list):
    """
    Exact 3-knot restricted cubic spline basis matching:
    Hmisc::rcspline.eval(x, knots=..., inclx=True, norm=2)

    Returns:
      b1 = x
      b2 = nonlinear basis
    """
    k1, k2, k3 = [float(v) for v in knots_list]

    b1 = float(x)

    b2 = (
        _pos_cube(x - k1)
        + (
            (k2 - k1) * _pos_cube(x - k3)
            - (k3 - k1) * _pos_cube(x - k2)
        ) / (k3 - k2)
    ) / ((k3 - k1) ** 2)

    return b1, b2

# =========================================
# 4) 从 cleaned predictors 构建设计矩阵 1 行
#    完全按 R 的 x_columns_os.csv 来
# =========================================
def normalize_cat(v):
    if pd.isna(v):
        raise ValueError(f"Missing categorical value: {v}")
    fv = float(v)
    if abs(fv - round(fv)) > 1e-12:
        raise ValueError(f"Categorical value is not integer-like: {v}")
    return str(int(round(fv)))

def build_design_row_cleaned(row: dict) -> pd.DataFrame:
    # 先准备一个 0 向量
    x = {col: 0.0 for col in x_col_order}

    # 连续变量
    x["Age"] = float(row["Age"])
    x["interval_post"] = float(row["interval_post"])
    x["LMR_pre_w"] = float(row["LMR_pre_w"])
    x["ALB_pre_w"] = float(row["ALB_pre_w"])
    x["ALB_L_w"] = float(row["ALB_L_w"])
    x["HB_pre_w"] = float(row["HB_pre_w"])

    # 分类变量 dummy（参考组不写，保持 0）
    p16 = normalize_cat(row["p16"])
    stage0 = normalize_cat(row["Stage0"])
    smoke = normalize_cat(row["Smoke"])

    if p16 == "1":
        x["p161"] = 1.0
    elif p16 == "2":
        x["p162"] = 1.0
    elif p16 != "0":
        raise ValueError(f"Invalid p16: {p16}")

    if stage0 == "2":
        x["Stage02"] = 1.0
    elif stage0 != "1":
        raise ValueError(f"Invalid Stage0: {stage0}")

    if smoke == "1":
        x["Smoke1"] = 1.0
    elif smoke == "2":
        x["Smoke2"] = 1.0
    elif smoke != "0":
        raise ValueError(f"Invalid Smoke: {smoke}")

    # RCS
    rcs_col1 = "rcs(LMR_dt_w, c(-0.088823529, -0.037810232, -0.008348214))LMR_dt_w"
    rcs_col2 = "rcs(LMR_dt_w, c(-0.088823529, -0.037810232, -0.008348214))LMR_dt_w'"

    b1, b2 = rcs_basis_3knots(float(row["LMR_dt_w"]), knots)
    x[rcs_col1] = b1
    x[rcs_col2] = b2

    return pd.DataFrame([x], columns=x_col_order)


# =========================================
# 5) 预测函数：输入 cleaned predictors，输出 lp / risk
# =========================================
def predict_fasti_os_cleaned(row: dict, times=(36.0, 60.0)) -> dict:
    X = build_design_row_cleaned(row)

    # 系数顺序：按 coef_os.csv 的 term
    beta = np.array([beta_map[t] for t in coef_df["term"]], dtype=float)

    # X 顺序：按 x_columns_os.csv，与 beta 对齐
    x_vec = X.iloc[0].to_numpy(dtype=float)

    lp_raw = float(np.dot(x_vec, beta))
    lp = lp_raw - center

    out = {
        "lp": lp,
    }

    for t in times:
        s0 = baseline_survival_at_time(float(t))
        risk_t = 1.0 - (s0 ** math.exp(lp))
        out[f"OSrisk{int(t)}"] = risk_t

    return out


# =========================================
# 6) 用 test_cases_os.csv 做一致性核对
# =========================================
rows = []
for i, r in test_df.iterrows():
    row_in = {
        "p16": r["p16"],
        "Stage0": r["Stage0"],
        "Age": float(r["Age"]),
        "Smoke": r["Smoke"],
        "interval_post": float(r["interval_post"]),
        "LMR_pre_w": float(r["LMR_pre_w"]),
        "ALB_pre_w": float(r["ALB_pre_w"]),
        "ALB_L_w": float(r["ALB_L_w"]),
        "HB_pre_w": float(r["HB_pre_w"]),
        "LMR_dt_w": float(r["LMR_dt_w"]),
    }

    pred = predict_fasti_os_cleaned(row_in, times=(36.0, 60.0))

    rows.append({
        "idx": i + 1,
        "lp_py": pred["lp"],
        "lp_r": float(r["lp_expected"]),
        "diff_lp": pred["lp"] - float(r["lp_expected"]),
        "risk36_py": pred["OSrisk36"],
        "risk36_r": float(r["OSrisk36_expected"]),
        "diff_risk36": pred["OSrisk36"] - float(r["OSrisk36_expected"]),
        "risk60_py": pred["OSrisk60"],
        "risk60_r": float(r["OSrisk60_expected"]),
        "diff_risk60": pred["OSrisk60"] - float(r["OSrisk60_expected"]),
    })

check_df = pd.DataFrame(rows)

print("=== check_df head ===")
print(check_df.head(10).to_string(index=False))

print("\n=== max abs diffs ===")
print({
    "max_abs_diff_lp": float(np.max(np.abs(check_df["diff_lp"]))),
    "max_abs_diff_risk36": float(np.max(np.abs(check_df["diff_risk36"]))),
    "max_abs_diff_risk60": float(np.max(np.abs(check_df["diff_risk60"]))),
})

# =========================================
# 7) Compare exact design rows column-by-column
# =========================================
x_r_df = pd.read_csv(f"{DEPLOY_DIR}/test_xrows_os.csv")

cmp_rows = []
for i, r in test_df.iterrows():
    row_in = {
        "p16": r["p16"],
        "Stage0": r["Stage0"],
        "Age": float(r["Age"]),
        "Smoke": r["Smoke"],
        "interval_post": float(r["interval_post"]),
        "LMR_pre_w": float(r["LMR_pre_w"]),
        "ALB_pre_w": float(r["ALB_pre_w"]),
        "ALB_L_w": float(r["ALB_L_w"]),
        "HB_pre_w": float(r["HB_pre_w"]),
        "LMR_dt_w": float(r["LMR_dt_w"]),
    }

    x_py = build_design_row_cleaned(row_in).iloc[0]
    x_r = x_r_df.iloc[i]

    for col in x_col_order:
        cmp_rows.append({
            "row_id": i + 1,
            "col": col,
            "x_py": float(x_py[col]),
            "x_r": float(x_r[col]),
            "diff": float(x_py[col] - x_r[col]),
        })

cmp_df = pd.DataFrame(cmp_rows)

print("\n=== worst design-matrix diffs ===")
print(
    cmp_df.reindex(cmp_df["diff"].abs().sort_values(ascending=False).index)
          .head(20)
          .to_string(index=False)
)

print("\n=== max abs design-matrix diff by column ===")
print(
    cmp_df.groupby("col", as_index=False)["diff"]
          .apply(lambda s: float(np.max(np.abs(s))))
)

# ===== Python: compare raw LP =====
lpraw_df = pd.read_csv(f"{DEPLOY_DIR}/test_lpraw_os.csv")
contrib1_r = pd.read_csv(f"{DEPLOY_DIR}/test_contrib_row1_os.csv")

# 重新算前10例 raw LP
lpraw_rows = []
for i, r in test_df.iterrows():
    row_in = {
        "p16": r["p16"],
        "Stage0": r["Stage0"],
        "Age": float(r["Age"]),
        "Smoke": r["Smoke"],
        "interval_post": float(r["interval_post"]),
        "LMR_pre_w": float(r["LMR_pre_w"]),
        "ALB_pre_w": float(r["ALB_pre_w"]),
        "ALB_L_w": float(r["ALB_L_w"]),
        "HB_pre_w": float(r["HB_pre_w"]),
        "LMR_dt_w": float(r["LMR_dt_w"]),
    }
    X1 = build_design_row_cleaned(row_in).iloc[0].to_numpy(dtype=float)
    beta = np.array([beta_map[t] for t in coef_df["term"]], dtype=float)
    lp_raw_py = float(np.dot(X1, beta))
    lp_py = lp_raw_py - center

    lpraw_rows.append({
        "row_id": i + 1,
        "lp_raw_py": lp_raw_py,
        "lp_raw_r": float(lpraw_df.loc[i, "lp_raw_expected"]),
        "diff_lp_raw": lp_raw_py - float(lpraw_df.loc[i, "lp_raw_expected"]),
        "lp_py": lp_py,
        "lp_r": float(lpraw_df.loc[i, "lp_expected"]),
        "diff_lp": lp_py - float(lpraw_df.loc[i, "lp_expected"]),
    })

lpraw_cmp = pd.DataFrame(lpraw_rows)

print("\n=== lpraw_cmp ===")
print(lpraw_cmp.to_string(index=False))

# 第1例逐项贡献
row1 = test_df.iloc[0]
row1_in = {
    "p16": row1["p16"],
    "Stage0": row1["Stage0"],
    "Age": float(row1["Age"]),
    "Smoke": row1["Smoke"],
    "interval_post": float(row1["interval_post"]),
    "LMR_pre_w": float(row1["LMR_pre_w"]),
    "ALB_pre_w": float(row1["ALB_pre_w"]),
    "ALB_L_w": float(row1["ALB_L_w"]),
    "HB_pre_w": float(row1["HB_pre_w"]),
    "LMR_dt_w": float(row1["LMR_dt_w"]),
}
x1_py = build_design_row_cleaned(row1_in).iloc[0]

contrib1_py = pd.DataFrame({
    "term": x_col_order,
    "x_py": [float(x1_py[c]) for c in x_col_order],
    "beta_py": [float(beta_map[t]) for t in coef_df["term"]],
})
contrib1_py["contrib_py"] = contrib1_py["x_py"] * contrib1_py["beta_py"]

contrib_cmp = contrib1_r.merge(contrib1_py, on="term", how="outer")
contrib_cmp["diff_x"] = contrib_cmp["x_py"] - contrib_cmp["x"]
contrib_cmp["diff_beta"] = contrib_cmp["beta_py"] - contrib_cmp["beta"]
contrib_cmp["diff_contrib"] = contrib_cmp["contrib_py"] - contrib_cmp["contrib"]

print("\n=== contrib_cmp_row1 ===")
print(contrib_cmp.to_string(index=False))
