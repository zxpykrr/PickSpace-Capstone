
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_absolute_percentage_error, make_scorer,
)
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, os.environ.get("DATA_DIR", "data_plus"))
OUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

X = pd.read_csv(f"{DATA_DIR}/X_features.csv")
y_df = pd.read_csv(f"{DATA_DIR}/y_target.csv")
ref = pd.read_csv(f"{DATA_DIR}/reference_ids.csv")

is_secondhand = (ref["property_id"] >= 1_000_000).values
print(f"ข้อมูลทั้งหมด: {(~is_secondhand).sum()} แถวจากโครงการ (เดิม) + {is_secondhand.sum()} แถวจากบ้านมือสอง (ใหม่)")

X["is_secondhand"] = is_secondhand.astype(int)

LOCATION_COORD_FEATURES = ["latitude", "longitude"]
X = X.drop(columns=LOCATION_COORD_FEATURES, errors="ignore")
print(f"ตัดฟีเจอร์พิกัดดิบออกตามการออกแบบ (ใช้ระยะห่างสถานที่/ถนนแทน): {LOCATION_COORD_FEATURES}")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area_per_bedroom"] = df["usable_area_sqm"] / df["bedrooms"].clip(lower=1)
    df["land_to_usable_ratio"] = df["land_area_sqw"] / df["usable_area_sqm"].clip(lower=1)
    df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"].clip(lower=1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["parking"]
    df["poi_concentration"] = df["poi_within_3km"] / df["poi_within_5km"].clip(lower=1)
    return df


ENGINEERED_FEATURES = [
    "area_per_bedroom", "land_to_usable_ratio", "bath_bed_ratio",
    "total_rooms", "poi_concentration",
]
X = add_engineered_features(X)
print(f"เพิ่มฟีเจอร์ใหม่จากอัตราส่วน/ผลรวมของฟีเจอร์เดิม: {ENGINEERED_FEATURES}")

ELEVATION_CSV = os.path.join(DATA_DIR, "elevation.csv")
if os.path.exists(ELEVATION_CSV):
    elev_df = pd.read_csv(ELEVATION_CSV)[["location_id", "elevation_m"]]
    elev_map = dict(zip(elev_df["location_id"], elev_df["elevation_m"]))
    elev_series = ref["location_id"].map(elev_map)
    n_missing = int(elev_series.isnull().sum())
    if n_missing:
        elev_series = elev_series.fillna(elev_series.median())
    X["elevation_m"] = elev_series.values
    print(f"เพิ่มฟีเจอร์ระดับความสูงภูมิประเทศ (elevation_m) — เติมค่ากลางให้ {n_missing} แถวที่ไม่มีข้อมูล")
else:
    print("ข้าม elevation_m: ไม่พบ data/elevation.csv (รัน node ml2/fetch_elevation.js เพื่อสร้าง)")

FLOOD_CSV = os.path.join(DATA_DIR, "flood_points.csv")
if os.path.exists(FLOOD_CSV) and os.path.exists(ELEVATION_CSV):
    flood_df = pd.read_csv(FLOOD_CSV)
    coords_df = pd.read_csv(ELEVATION_CSV)
    coord_map = {r.location_id: (r.latitude, r.longitude) for r in coords_df.itertuples()}

    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    row_coords = ref["location_id"].map(coord_map)
    lat_arr = np.array([c[0] if isinstance(c, tuple) else np.nan for c in row_coords], dtype=float)
    lon_arr = np.array([c[1] if isinstance(c, tuple) else np.nan for c in row_coords], dtype=float)

    def _nearest_dist(points_df):
        stack = np.column_stack([
            _haversine_m(lat_arr, lon_arr, p.latitude, p.longitude) for p in points_df.itertuples()
        ])
        d = np.nanmin(stack, axis=1)
        return np.where(np.isnan(d), np.nanmedian(d), d)

    risky_df = flood_df[flood_df["flood_risk_level"].isin(["high", "medium"])]
    safe_df = flood_df[flood_df["flood_risk_level"] == "safe"]
    n_no_coord = int(np.isnan(lat_arr).sum())

    if len(risky_df) == 0:
        print("ข้ามฟีเจอร์น้ำท่วม: flood_points.csv ไม่มีจุด high/medium เลย")
    else:
        X["dist_flood_m"] = _nearest_dist(risky_df)
        msg = (f"เพิ่มฟีเจอร์น้ำท่วมจาก {len(risky_df)} จุดเสี่ยง"
               f" + {len(safe_df)} จุดปลอดภัย (เติมค่ากลางให้ {n_no_coord} แถวที่พิกัดเสีย)")
        if len(safe_df) > 0:
            X["dist_safe_m"] = _nearest_dist(safe_df)
            X["flood_risk_score"] = ((X["dist_safe_m"] - X["dist_flood_m"])
                                     / (X["dist_safe_m"] + X["dist_flood_m"]).clip(lower=1))
        else:
            msg += " — ไม่มีจุด safe จึงได้แค่ dist_flood_m (เสี่ยงเป็นตัวแทนระยะห่างใจกลางเมือง)"
        print(msg)
else:
    print("ข้ามฟีเจอร์น้ำท่วม: ต้องมีทั้ง data/flood_points.csv และ data/elevation.csv")

APPRAISAL_JSON = os.path.join(OUT_DIR, "government_appraisal.json")
_loc_names = pd.read_csv(os.path.join(DATA_DIR, "dim_location.csv"))[["location_id", "location_name"]]
LOCATION_NAME_MAP = dict(zip(_loc_names["location_id"], _loc_names["location_name"]))
zone_land_appraisal = {}
if os.path.exists(APPRAISAL_JSON):
    with open(APPRAISAL_JSON, encoding="utf-8") as f:
        gov_appraisal = json.load(f)
    land_by_tambon = gov_appraisal.get("land_appraisal_by_tambon", {})

    def _tambon_of(location_id):
        name = LOCATION_NAME_MAP.get(location_id)
        return name.split(" - ")[-1].strip() if isinstance(name, str) else None

    row_tambon = ref["location_id"].map(_tambon_of)
    land_price = row_tambon.map(
        lambda t: land_by_tambon.get(t, {}).get("price_per_sqw_current") if t else None
    )
    n_unmatched = int(land_price.isnull().sum())
    land_price = land_price.fillna(land_price.median()).astype(float)
    X["land_appraisal_per_sqw"] = land_price.values

    for loc_id, name in LOCATION_NAME_MAP.items():
        if not isinstance(name, str):
            continue
        t = name.split(" - ")[-1].strip()
        v = land_by_tambon.get(t, {}).get("price_per_sqw_current")
        if v is not None:
            zone_land_appraisal[name] = float(v)

    print(f"เพิ่มฟีเจอร์ราคาประเมินที่ดินรายตำบล (land_appraisal_per_sqw) "
          f"— จับคู่ได้ {len(land_price) - n_unmatched}/{len(land_price)} แถว, "
          f"ถูกสุด {land_price.min():,.0f} ถึงแพงสุด {land_price.max():,.0f} บาท/ตร.วา")
else:
    print("ข้าม land_appraisal_per_sqw: ไม่พบ artifacts/government_appraisal.json")

LOCATION_CSV = os.path.join(DATA_DIR, "dim_location.csv")
zone_names_df = pd.read_csv(LOCATION_CSV)[["location_id", "location_name"]]


groups = ref["group_id"] if "group_id" in ref.columns else ref["property_id"]
if "group_id" not in ref.columns:
    print("เตือน: reference_ids.csv ยังไม่มีคอลัมน์ group_id — ใช้ property_id แทน "
          "(รัน build_secondhand_training_data.js ใหม่เพื่อสร้าง group_id ที่กัน leakage ได้ถูกต้อง)")

SELECTED_FEATURES = [
    "usable_area_sqm", "land_area_sqw", "bathrooms", "parking", "floors",
    "total_rooms", "area_per_bedroom", "land_to_usable_ratio", "bath_bed_ratio",
    "dist_ห้างสรรพสินค้า_m", "dist_โรงพยาบาล_m", "dist_flood_m",
    "elevation_m", "topsis_score",
]
_missing = [c for c in SELECTED_FEATURES if c not in X.columns]
if _missing:
    raise SystemExit(f"ไม่พบฟีเจอร์ที่ต้องใช้: {_missing}")
_dropped = [c for c in X.columns if c not in SELECTED_FEATURES]
X_ALL = X.copy()
X = X[SELECTED_FEATURES]
print(f"คัดเหลือ {len(SELECTED_FEATURES)} ฟีเจอร์ที่ผ่านการคัดด้วย CV (ตัดออก {len(_dropped)} ตัว)")

print(f"จำนวนตัวอย่าง: {len(X)} | จำนวน feature: {X.shape[1]}")

_export = X.copy()
_export["starting_price_thb"] = y_df["starting_price_thb"].values
_export["group_id"] = groups.values
_export.to_csv(f"{DATA_DIR}/dataset_full.csv", index=False, encoding="utf-8-sig")
print(f"  export ชุดข้อมูลสำเร็จรูป -> data/dataset_full.csv ({len(_export)} แถว x {_export.shape[1]} คอลัมน์)")

X_primary = X[~is_secondhand]
y_price_primary = y_df["starting_price_thb"].values[~is_secondhand]
groups_primary = groups[~is_secondhand]
gss_reg = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx_primary, test_idx_reg = next(gss_reg.split(X_primary, y_price_primary, groups_primary))
X_test_reg = X_primary.iloc[test_idx_reg]
y_price_test = y_price_primary[test_idx_reg]

test_groups_reg = set(groups_primary.iloc[test_idx_reg])
primary_row_pos = np.where(~is_secondhand)[0]
secondhand_row_pos = np.where(is_secondhand)[0]
secondhand_safe = np.array(
    [p for p in secondhand_row_pos if groups.iloc[p] not in test_groups_reg], dtype=int
)
n_dropped = len(secondhand_row_pos) - len(secondhand_safe)
train_row_pos = np.concatenate([primary_row_pos[train_idx_primary], secondhand_safe])
X_train_reg = X.iloc[train_row_pos]
y_price_train = y_df["starting_price_thb"].values[train_row_pos]
groups_train_reg = groups.iloc[train_row_pos]
print(f"Regression เทรนด้วยมือ1+มือ2 ({len(X_train_reg)} แถว) | วัดผลด้วยมือ1 held-out เท่านั้น ({len(X_test_reg)} แถว)")
print(f"  ตัดบ้านมือสองที่อยู่โครงการเดียวกับ unit ใน test ออกจาก train: {n_dropped} แถว (กัน leakage)")


SECONDHAND_WEIGHT = 1.0
sample_weight_train = np.where(is_secondhand[train_row_pos], SECONDHAND_WEIGHT, 1.0)
print(f"  น้ำหนักแถวบ้านมือสอง x{SECONDHAND_WEIGHT:g} (ข้อมูลสองตลาดสมดุลแล้ว ไม่ต้องถ่วง)")


TARGET_POWER = 0.6


def to_target(y):
    """แปลงราคา (บาท) -> สเกลที่ใช้เทรน"""
    return np.log(y) if TARGET_POWER == 0 else np.asarray(y, dtype=float) ** TARGET_POWER


def from_target(t):
    """แปลงกลับจากสเกลที่เทรน -> ราคา (บาท)"""
    if TARGET_POWER == 0:
        return np.exp(t)
    return np.clip(t, 1e-9, None) ** (1.0 / TARGET_POWER)


def _mae_baht(y_true_t, y_pred_t):
    """MAE ในหน่วยบาทจริง แม้ตอน CV จะทำงานบนสเกลที่แปลงแล้ว — ให้เลข [tune] อ่านรู้เรื่อง"""
    return mean_absolute_error(from_target(y_true_t), from_target(y_pred_t))


MAE_BAHT_SCORER = make_scorer(_mae_baht, greater_is_better=False)

y_price_train_t = to_target(y_price_train)


def tune_reg(name, estimator, param_dist, n_iter=25):
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=n_iter,
        cv=GroupKFold(n_splits=5), scoring=MAE_BAHT_SCORER,
        random_state=42, n_jobs=-1,
    )
    search.fit(X_train_reg, y_price_train_t, groups=groups_train_reg,
               sample_weight=sample_weight_train)
    print(f"  [tune] {name}: best CV MAE={-search.best_score_:,.0f} บาท  params={search.best_params_}")
    return search.best_estimator_


print("กำลังหา hyperparameter ที่ดีที่สุดของแต่ละโมเดล regression (ค้นหาบน train set เท่านั้น)...")
regressors = {
    "Random Forest": tune_reg(
        "Random Forest",
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [6, 10, 15, None],
            "min_samples_leaf": [1, 2, 3, 5],
            "max_features": [0.3, 0.5, 0.7, "sqrt"],
        },
    ),
    "LightGBM": tune_reg(
        "LightGBM",
        LGBMRegressor(random_state=42, verbose=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, -1],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [3, 5, 10],
            "learning_rate": [0.03, 0.05, 0.1],
        },
    ),
    "Gradient Boosting": tune_reg(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [1, 3, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
    ),
    "CatBoost": tune_reg(
        "CatBoost",
        CatBoostRegressor(random_state=42, verbose=False),
        {
            "iterations": [100, 200, 300],
            "depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 8],
        },
    ),
}

regressors["Voting Ensemble"] = VotingRegressor(
    estimators=[
        ("rf", regressors["Random Forest"]),
        ("lgbm", regressors["LightGBM"]),
        ("gb", regressors["Gradient Boosting"]),
        ("cat", regressors["CatBoost"]),
    ],
)
print()

regression_results = {}
print(f"\n{'='*50}")
print("=== เปรียบเทียบโมเดล Regression (ทำนายราคาเป็นตัวเลข) ===")
for name, model in regressors.items():
    model.fit(X_train_reg, y_price_train_t, sample_weight=sample_weight_train)
    pred = from_target(model.predict(X_test_reg))
    m = mean_absolute_error(y_price_test, pred)
    r = r2_score(y_price_test, pred)
    mp = mean_absolute_percentage_error(y_price_test, pred)
    regression_results[name] = {"mae": float(m), "r2": float(r), "mape": float(mp)}
    print(f"  {name}: MAE={m:,.0f} บาท, R^2={r:.3f}, MAPE={mp * 100:.1f}%")


best_regressor_name = min(regression_results, key=lambda k: regression_results[k]["mae"])
regressor = regressors[best_regressor_name]
mae = regression_results[best_regressor_name]["mae"]
r2 = regression_results[best_regressor_name]["r2"]
mape = regression_results[best_regressor_name]["mape"]

print()
print(f"โมเดล Regression ที่ดีที่สุด: {best_regressor_name}")
print(f"  Test set -> MAE={mae:,.0f} บาท, R^2={r2:.3f}, MAPE={mape * 100:.1f}%")
print("หมายเหตุ: ใช้เป็น 'ราคาโดยประมาณ' พร้อมช่วงราคาที่คำนวณจาก MAPE "
      "ไม่ใช่ตัวเลขที่แม่นยำระดับบาทต่อบาท เพราะ MAE ยังสูงเมื่อเทียบกับราคาบ้านทั่วไป")
print("ใช้ MAPE (% คลาดเคลื่อนเฉลี่ย) แทน MAE (บาทคงที่) สำหรับคำนวณช่วงราคาที่แสดงในเว็บ "
      "เพราะ MAE คงที่ทำให้บ้านราคาถูกได้ช่วงราคากว้างเกินจริงเทียบเป็นสัดส่วน (เคยสูงถึง 70% "
      "ของราคาบ้านสำหรับบ้านราคาต่ำ) ส่วนบ้านราคาแพงกลับดูแคบเกินจริง")

r2_train = r2_score(y_price_train, from_target(regressor.predict(X_train_reg)))
print(f"  ส่วนต่าง R^2 train - test = {r2_train:.3f} - {r2:.3f} = {r2_train - r2:.3f} "
      f"({'ไม่โอเวอร์ฟิต' if r2_train - r2 <= 0.15 else 'เสี่ยงโอเวอร์ฟิต'} ตามเกณฑ์ไม่เกิน 0.15)")

PERM_REPEATS = 10


def _perm_importance(model, X_eval, y_eval_t):
    """MAE (บาท) ที่เพิ่มขึ้นเมื่อสลับค่าฟีเจอร์นั้น — ยิ่งมากยิ่งสำคัญ, <= 0 คือไม่ช่วยเลย"""
    return permutation_importance(
        model, X_eval, y_eval_t, scoring=MAE_BAHT_SCORER,
        n_repeats=PERM_REPEATS, random_state=42, n_jobs=-1,
    ).importances


def _cv_folds(cols):
    """GroupKFold ภายในชุดเทรนเท่านั้น — คืนโมเดลที่เทรนแต่ละ fold พร้อมข้อมูล validation"""
    for tr, va in GroupKFold(n_splits=5).split(X_train_reg, y_price_train_t, groups_train_reg):
        m = clone(regressor).fit(X_train_reg.iloc[tr][cols], y_price_train_t[tr],
                                 sample_weight=sample_weight_train[tr])
        yield m, X_train_reg.iloc[va][cols], y_price_train_t[va]


def _cv_mae(cols):
    return float(np.mean([_mae_baht(yv, m.predict(Xv)) for m, Xv, yv in _cv_folds(cols)]))


print()
print("กำลังวัด Permutation Importance (ชุดทดสอบ = รายงาน, CV ในชุดเทรน = ใช้ตัดสินใจ)")
all_cols = list(X.columns)
perm_test = _perm_importance(regressor, X_test_reg, to_target(y_price_test))
perm_folds = np.column_stack([
    _perm_importance(m, Xv, yv).mean(axis=1) for m, Xv, yv in _cv_folds(all_cols)
])
perm_cv_mean = perm_folds.mean(axis=1)
perm_cv_lower = perm_cv_mean - 1.96 * perm_folds.std(axis=1, ddof=1) / np.sqrt(perm_folds.shape[1])

perm_df = pd.DataFrame({
    "feature": all_cols,
    "perm_test_mae_increase": perm_test.mean(axis=1),
    "perm_test_std": perm_test.std(axis=1),
    "perm_cv_mae_increase": perm_cv_mean,
    "perm_cv_lower95": perm_cv_lower,
}).sort_values("perm_cv_mae_increase", ascending=False).reset_index(drop=True)

print(f"  {'ฟีเจอร์':<24}{'CV ในชุดเทรน (บาท)':>22}{'ขอบล่าง 95%':>14}{'ชุดทดสอบ (บาท)':>20}")
for row in perm_df.itertuples():
    flag = "  <- ยังพิสูจน์ไม่ได้ว่าช่วย" if row.perm_cv_lower95 <= 0 else ""
    print(f"  {row.feature:<24}{row.perm_cv_mae_increase:>22,.0f}{row.perm_cv_lower95:>14,.0f}"
          f"{row.perm_test_mae_increase:>12,.0f} ± {row.perm_test_std:>6,.0f}{flag}")
print("  ค่า = MAE ที่แย่ลงเมื่อสลับฟีเจอร์นั้น  ถ้าฟีเจอร์สองตัวสัมพันธ์กันสูง "
      "(เช่น total_rooms กับ bathrooms) คะแนนจะถูกแบ่งกันจนดูต่ำกว่าความจริง")

drop_candidates = perm_df.loc[perm_df["perm_cv_lower95"] <= 0, "feature"].tolist()
drop_check = []
if drop_candidates:
    cv_mae_base = _cv_mae(all_cols)
    print(f"\n  ทดลองตัดฟีเจอร์ที่น่าสงสัยทีละตัว (CV MAE ในชุดเทรน ปัจจุบัน {cv_mae_base:,.0f} บาท)")
    for f in drop_candidates:
        cv_mae_drop = _cv_mae([c for c in all_cols if c != f])
        delta = cv_mae_drop - cv_mae_base
        drop_check.append({"feature": f, "cv_mae_without": cv_mae_drop, "delta_mae": delta})
        verdict = "ตัดได้ (ไม่แย่ลง)" if delta <= 0 else "ควรเก็บไว้"
        print(f"    ตัด {f:<24} CV MAE = {cv_mae_drop:>9,.0f} บาท ({delta:+,.0f})  -> {verdict}")
    print("  ถ้าจะตัดจริง: แก้ SELECTED_FEATURES แล้วรันใหม่ และดูว่าคะแนนชุดทดสอบไม่แย่ลง")
else:
    print("\n  ทุกฟีเจอร์ช่วยลด MAE อย่างมีนัยสำคัญใน CV — ไม่มีตัวที่ควรตัด")

regressor.fit(X, to_target(y_df["starting_price_thb"].values),
              sample_weight=np.where(is_secondhand, SECONDHAND_WEIGHT, 1.0))

def _market_metrics(mask, seeds=(0, 1, 2, 3, 4)):
    Xg, pg, gg = X[mask], y_df["starting_price_thb"].values[mask], groups.values[mask]
    maes, mapes, r2s = [], [], []
    for sd in seeds:
        _, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=sd).split(Xg, pg, gg))
        test_groups = set(gg[te])
        tm = np.array([gi not in test_groups for gi in groups.values])
        m = clone(regressor).fit(
            X[tm], to_target(y_df["starting_price_thb"].values[tm]),
            sample_weight=np.where(is_secondhand[tm], SECONDHAND_WEIGHT, 1.0))
        pred, yt = from_target(m.predict(Xg.iloc[te])), pg[te]
        maes.append(mean_absolute_error(yt, pred))
        mapes.append(mean_absolute_percentage_error(yt, pred))
        r2s.append(r2_score(yt, pred))
    return {"mae": float(np.mean(maes)), "mape": float(np.mean(mapes)),
            "r2": float(np.mean(r2s)), "n": int(mask.sum())}


print()
print("กำลังวัดความแม่นแยกตลาด (มือ1 / มือ2)")
metrics_by_market = {
    "new": _market_metrics(~is_secondhand),
    "secondhand": _market_metrics(is_secondhand),
}
for key, label in (("new", "มือ1 (โครงการใหม่)"), ("secondhand", "มือ2 (บ้านมือสอง)")):
    mm = metrics_by_market[key]
    print(f"  {label:<20} n={mm['n']:>3}  MAE={mm['mae']:>9,.0f} บาท  "
          f"MAPE={mm['mape'] * 100:>5.1f}%  R^2={mm['r2']:>6.3f}")
print("  ช่วงราคาที่แสดงในเว็บจะใช้ MAPE ของตลาดที่ผู้ใช้เลือก ไม่ใช่ค่าเฉลี่ยรวม")

df_full = pd.concat([X_ALL.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
df_full["house_type"] = "บ้านเดี่ยว"
df_full.loc[df_full["house_type_คอนโด"] == 1, "house_type"] = "คอนโด"
df_full.loc[df_full["house_type_บ้านแฝด"] == 1, "house_type"] = "บ้านแฝด"

print(f"\nตัด unit ที่ year_start_be=0 ออกจากค่าเฉลี่ยปีโครงการ: {(df_full['year_start_be']==0).sum()} unit (ข้อมูลผิดปกติ)")
market_stats = {}
for house_type, group in df_full.groupby("house_type"):
    prices = group["starting_price_thb"]
    valid_area = group[group["usable_area_sqm"] > 0]
    price_per_sqm = (valid_area["starting_price_thb"] / valid_area["usable_area_sqm"])

    valid_year = group[group["year_start_be"] > 0]
    market_stats[house_type] = {
        "count": int(len(prices)),
        "mean": float(prices.mean()),
        "min": float(prices.min()),
        "max": float(prices.max()),
        "p10": float(prices.quantile(0.10)),
        "p25": float(prices.quantile(0.25)),
        "p50": float(prices.quantile(0.50)),
        "p75": float(prices.quantile(0.75)),
        "p90": float(prices.quantile(0.90)),
        "price_per_sqm_mean": float(price_per_sqm.mean()) if len(price_per_sqm) > 0 else None,
        "usable_area_sqm_mean": float(group["usable_area_sqm"].mean()),
        "land_area_sqw_mean": float(group["land_area_sqw"].mean()),
        "bedrooms_mean": float(group["bedrooms"].mean()),
        "bathrooms_mean": float(group["bathrooms"].mean()),
        "parking_mean": float(group["parking"].mean()),
        "year_start_be_mean": float(valid_year["year_start_be"].mean()) if len(valid_year) > 0 else None,
        "topsis_score_mean": float(group["topsis_score"].mean()),
        "distance_means": {
            col: float(group[col].mean())
            for col in [
                "dist_ตลาด_m", "dist_ถนนซอย_m", "dist_ถนนสายหลัก_m", "dist_มหาวิทยาลัย_m",
                "dist_สถานีขนส่ง_m", "dist_สถานีตำรวจ_m", "dist_สถานีรถไฟ_m", "dist_สนามบิน_m",
                "dist_สวนสาธารณะ_m", "dist_ห้างสรรพสินค้า_m", "dist_โรงงานผลิต_m",
                "dist_โรงงานอุตสาหกรรม_m", "dist_โรงพยาบาล_m", "dist_โรงเรียน_m",
            ]
        },
    }

print("\nสถิติตลาดแยกตามประเภท:")
for ht, stats in market_stats.items():
    print(f"  {ht}: n={stats['count']}, เฉลี่ย={stats['mean']:,.0f}, "
          f"ต่ำสุด={stats['min']:,.0f}, สูงสุด={stats['max']:,.0f}, "
          f"บ./ตรม.={stats['price_per_sqm_mean']:,.0f}")

df_zone = pd.concat([ref[["location_id"]].reset_index(drop=True), df_full.reset_index(drop=True)], axis=1)
df_zone = df_zone.merge(zone_names_df, on="location_id", how="left")

REIC_PRICE_BANDS = [
    (1_000_000, "ไม่เกิน 1.00 ล้านบาท"),
    (1_500_000, "1.01–1.50 ล้านบาท"),
    (2_000_000, "1.51–2.00 ล้านบาท"),
    (3_000_000, "2.01–3.00 ล้านบาท"),
    (5_000_000, "3.01–5.00 ล้านบาท"),
    (7_500_000, "5.01–7.50 ล้านบาท"),
    (10_000_000, "7.51–10.00 ล้านบาท"),
]
REIC_TOP_BAND = "มากกว่า 10.00 ล้านบาท"


def reic_price_band(price):
    for upper, label in REIC_PRICE_BANDS:
        if price <= upper:
            return label
    return REIC_TOP_BAND


zone_stats = {}
for zone_name, zgroup in df_zone.groupby("location_name"):
    if pd.isna(zone_name):
        continue
    zone_entry = {"total_count": int(len(zgroup)), "by_type": {}}
    for ht, tgroup in zgroup.groupby("house_type"):
        prices = tgroup["starting_price_thb"]
        zone_entry["by_type"][ht] = {
            "count": int(len(prices)),
            "mean": float(prices.mean()),
            "tier": reic_price_band(prices.mean()),
        }
    zone_stats[zone_name] = zone_entry

print(f"\nจำนวนทำเลทั้งหมดที่มีข้อมูล: {len(zone_stats)}")
print(f"ทำเลที่มีข้อมูล >= 5 unit: {sum(1 for z in zone_stats.values() if z['total_count'] >= 5)}")

zone_elevation = {}
if "elevation_m" in X.columns:
    df_zone_elev = df_zone.copy()
    df_zone_elev["elevation_m"] = X["elevation_m"].values
    for zone_name, zgroup in df_zone_elev.groupby("location_name"):
        if pd.isna(zone_name):
            continue
        zone_elevation[zone_name] = float(zgroup["elevation_m"].median())
    print(f"ระดับความสูงเฉลี่ยรายทำเล: {len(zone_elevation)} ทำเล "
          f"(ต่ำสุด {min(zone_elevation.values()):.0f} ม. สูงสุด {max(zone_elevation.values()):.0f} ม.)")

FLOOD_FEATURES = [c for c in ["dist_flood_m", "dist_safe_m", "flood_risk_score"] if c in X.columns]
zone_flood = {}
if FLOOD_FEATURES:
    df_zone_flood = df_zone.copy()
    for c in FLOOD_FEATURES:
        df_zone_flood[c] = X[c].values
    for zone_name, zgroup in df_zone_flood.groupby("location_name"):
        if pd.isna(zone_name):
            continue
        zone_flood[zone_name] = {c: float(zgroup[c].median()) for c in FLOOD_FEATURES}
    dists = [v["dist_flood_m"] for v in zone_flood.values()]
    print(f"ค่าน้ำท่วมรายทำเล ({', '.join(FLOOD_FEATURES)}): {len(zone_flood)} ทำเล "
          f"(ระยะจุดเสี่ยง ใกล้สุด {min(dists):.0f} ม. ไกลสุด {max(dists):.0f} ม.)")

comparable_cols = ["usable_area_sqm", "land_area_sqw", "bedrooms", "bathrooms",
                    "parking", "starting_price_thb", "house_type", "is_secondhand",
                    "location_name", "latitude", "longitude"]
_coords = (pd.read_csv(ELEVATION_CSV)[["location_id", "latitude", "longitude"]]
           if os.path.exists(ELEVATION_CSV)
           else pd.DataFrame(columns=["location_id", "latitude", "longitude"]))
_comp_loc = df_zone[["location_id", "location_name"]].merge(_coords, on="location_id", how="left")
df_comp = df_full.copy()
for c in ("location_name", "latitude", "longitude"):
    df_comp[c] = _comp_loc[c].values
df_comp[["latitude", "longitude"]] = df_comp[["latitude", "longitude"]].astype(object).where(
    df_comp[["latitude", "longitude"]].notna(), None)
df_comp["location_name"] = df_comp["location_name"].where(df_comp["location_name"].notna(), None)
n_before = len(df_full)
comparables_source = df_comp[df_comp["usable_area_sqm"] > 0]
print(f"\nตัด unit ที่ usable_area_sqm=0 ออกจาก comparables: {n_before - len(comparables_source)} unit")
comparables_data = comparables_source[comparable_cols].to_dict(orient="records")
_n_2h = sum(1 for d in comparables_data if d["is_secondhand"] == 1)
print(f"comparables: {len(comparables_data)} รายการ (มือ1 {len(comparables_data) - _n_2h} · มือ2 {_n_2h})")
_n_no_coord = sum(1 for d in comparables_data if d["latitude"] is None)
print(f"  มีทำเล+พิกัด {len(comparables_data) - _n_no_coord} รายการ (ไม่มีพิกัด {_n_no_coord})")
joblib.dump({
    "regressor": regressor,
    "feature_columns": list(X.columns),
    "market_stats": market_stats,
    "zone_stats": zone_stats,
    "zone_elevation": zone_elevation,
    "zone_flood": zone_flood,
    "zone_land_appraisal": zone_land_appraisal,
    "comparables_data": comparables_data,
    "target_transform": {"kind": "power", "power": float(TARGET_POWER)},
    "regression_metrics": {"mae": float(mae), "r2": float(r2), "mape": float(mape)},
    "regression_metrics_by_market": metrics_by_market,
}, f"{SCRIPT_DIR}/model.pkl")

metrics_summary = {
    "train_test_split": {
        "train_size": len(X_train_reg),
        "test_size": len(X_test_reg),
    },
    "best_regressor": best_regressor_name,
    "regression_metrics_by_market": metrics_by_market,
    "regression_comparison": regression_results,
    "overfit_check": {"r2_train": float(r2_train), "r2_test": float(r2),
                      "r2_gap": float(r2_train - r2)},
    "permutation_importance": perm_df.to_dict(orient="records"),
    "permutation_drop_check": drop_check,
}
with open(f"{OUT_DIR}/metrics_report.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

def get_feature_importances(model):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            imp = get_feature_importances(step)
            if imp is not None:
                return imp
    if hasattr(model, "estimators_"):
        importances = [get_feature_importances(est) for est in model.estimators_]
        importances = [imp for imp in importances if imp is not None]
        if importances:
            return np.mean(importances, axis=0)
    return None


best_model_importances = get_feature_importances(regressor)
if best_model_importances is not None:
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model_importances
    }).sort_values("importance", ascending=False)
    fi.to_csv(f"{OUT_DIR}/feature_importance_mdi.csv", index=False, encoding="utf-8-sig")
    print("\nFeature importance แบบ MDI (top 10) — วัดจากชุดเทรน เก็บไว้เทียบเท่านั้น:")
    print(fi.head(10).to_string(index=False))

    cmp = fi.rename(columns={"importance": "mdi"}).merge(perm_df, on="feature")
    cmp["rank_mdi"] = cmp["mdi"].rank(ascending=False).astype(int)
    cmp["rank_perm_cv"] = cmp["perm_cv_mae_increase"].rank(ascending=False).astype(int)
    cmp["rank_perm_test"] = cmp["perm_test_mae_increase"].rank(ascending=False).astype(int)
    cmp.to_csv(f"{OUT_DIR}/feature_importance_compare.csv", index=False, encoding="utf-8-sig")
    print("\nเทียบอันดับ MDI กับ Permutation (CV / ชุดทดสอบ):")
    print(cmp[["feature", "rank_mdi", "rank_perm_cv", "rank_perm_test"]].to_string(index=False))

perm_share = perm_df["perm_cv_mae_increase"].clip(lower=0)
pd.DataFrame({
    "feature": perm_df["feature"],
    "importance": perm_share / perm_share.sum(),
}).sort_values("importance", ascending=False).to_csv(
    f"{OUT_DIR}/feature_importance.csv", index=False, encoding="utf-8-sig")

print(f"\nบันทึกไฟล์โมเดลที่: {SCRIPT_DIR}/model.pkl (regressor + feature_columns + สถิติ รวมไฟล์เดียว)")
print(f"บันทึกรายงานผล/สถิติเพิ่มเติมที่: {OUT_DIR}/ (metrics_report.json, feature_importance.csv, ฯลฯ)")
print("  - metrics_report.json (ผลเปรียบเทียบโมเดล regression + permutation importance)")
print("  - feature_importance.csv (Permutation — หน้าเว็บใช้ไฟล์นี้)")
print("  - feature_importance_mdi.csv (MDI แบบเดิม เก็บไว้เทียบ)")
print("  - feature_importance_compare.csv (อันดับ MDI เทียบ Permutation)")
print("\nกำลังประเมินราคาบ้านในระบบแบบ out-of-fold (ทายด้วยโมเดลที่ไม่เคยเห็นโครงการนั้น)...")
y_all = y_df["starting_price_thb"].values
oof_pred = np.full(len(X), np.nan)
for tr, te in GroupKFold(n_splits=5).split(X, y_all, groups):
    _m = clone(regressor).fit(X.iloc[tr], to_target(y_all[tr]),
                              sample_weight=np.where(is_secondhand[tr], SECONDHAND_WEIGHT, 1.0))
    oof_pred[te] = from_target(_m.predict(X.iloc[te]))

valid_rows = (X_ALL["usable_area_sqm"] > 0).values
oof_by_market = {}
for key, mask in (("new", ~is_secondhand), ("secondhand", is_secondhand)):
    mk = mask & valid_rows
    oof_by_market[key] = {
        "mape": float(mean_absolute_percentage_error(y_all[mk], oof_pred[mk])),
        "mae": float(mean_absolute_error(y_all[mk], oof_pred[mk])),
        "n": int(mk.sum()),
    }
    print(f"  {key:<11} n={oof_by_market[key]['n']:>4}  MAPE={oof_by_market[key]['mape'] * 100:5.1f}%  "
          f"MAE={oof_by_market[key]['mae']:>11,.0f} บาท")


def _db_ref(unit_id, secondhand):
    """แปลง unit_id ในชุดเทรนกลับเป็น id ในฐานข้อมูล — แถวที่ดึงจาก nayoo (9,000,000 ขึ้นไป) ไม่มีในฐานข้อมูล"""
    if not secondhand:
        return ("unit", unit_id) if unit_id < 1_000_000 else None
    if 1_000_000 <= unit_id < 2_000_000:
        return ("secondhand", unit_id - 1_000_000)
    if 2_000_000 <= unit_id < 3_000_000:
        return ("secondhand", unit_id - 2_000_000)
    return None


import predict as _predict

listing_rows = []
for i in range(len(X_ALL)):
    if not valid_rows[i]:
        continue
    ref_db = _db_ref(int(ref["unit_id"].iloc[i]), bool(is_secondhand[i]))
    if ref_db is None:
        continue
    house_type = df_full["house_type"].iloc[i]
    zone = df_zone["location_name"].iloc[i]
    zone = zone if isinstance(zone, str) else None
    est = float(oof_pred[i])
    actual = float(y_all[i])
    payload = {
        "floors": float(X_ALL["floors"].iloc[i]),
        "land_area_sqw": float(X_ALL["land_area_sqw"].iloc[i]),
        "usable_area_sqm": float(X_ALL["usable_area_sqm"].iloc[i]),
    }
    gf = _predict.compute_government_forecast(payload, house_type, zone, est) if zone else None
    listing_rows.append({
        "source": ref_db[0],
        "id": int(ref_db[1]),
        "house_type": house_type,
        "is_secondhand": int(is_secondhand[i]),
        "zone": zone,
        "usable_area_sqm": payload["usable_area_sqm"],
        "land_area_sqw": payload["land_area_sqw"],
        "bedrooms": int(X_ALL["bedrooms"].iloc[i]),
        "bathrooms": int(X_ALL["bathrooms"].iloc[i]),
        "price_actual": round(actual),
        "price_estimate": round(est),
        "diff_pct": round((actual - est) / est * 100, 1) if est > 0 else None,
        "annual_growth_pct": gf["annual_growth_pct"] if gf else None,
        "projected_next_year": gf["market_price_projection"]["projected_next_year"] if gf else None,
    })

with open(f"{OUT_DIR}/listing_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "method": "out-of-fold (GroupKFold 5 ส่วน แบ่งตามโครงการ)",
        "model": best_regressor_name,
        "accuracy_by_market": oof_by_market,
        "listings": listing_rows,
    }, f, ensure_ascii=False)
_n_sh = sum(r["is_secondhand"] for r in listing_rows)
print(f"  บันทึก listing_predictions.json: {len(listing_rows)} หลังที่อยู่ในฐานข้อมูล "
      f"(มือ1 {len(listing_rows) - _n_sh} · มือ2 {_n_sh}) — ตัดแถวที่ดึงจาก nayoo และพื้นที่ใช้สอย 0 ออก")
