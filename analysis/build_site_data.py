#!/usr/bin/env python
"""
Build the derived data files used by the Southern California El Nino website (index.html).

Usage (from the project directory):
    python -I analysis/build_site_data.py ~/Downloads/ScalENSO data

Raw inputs (downloaded into RAW_DIR):
  enso_indices/RONI.ascii.txt, oni.ascii.txt, wksst9120.for, ersst.v5.pdo.dat   NOAA CPC / NCEI
  ersst/sst.mnmean.nc                      NOAA ERSSTv5 (PSL copy)
  nclimdiv/climdiv-pcpndv-v1.0.0-*         NOAA nClimDiv divisional precipitation
  nclimdiv/shp/GIS.OFFICIAL_CLIM_DIVISIONS.shp
  ghcnd/by_station/*.csv.gz, ghcnd/ghcnd-stations.txt   GHCN-Daily
  forecasts/cpc_*roni_outlook*.html, *roni_strengths*.html, *roni_probabilities*.html
  naturalearth/ne_110m_land.geojson

Outputs (in OUT_DIR):
  scal_data.js  - indices, precipitation, model ensembles, hindcasts, composites (window.SCAL_DATA)
  geo_data.js   - climate-division map and SST anomaly maps (window.SCAL_GEO)
  *.csv         - tidy tables offered for download
"""
import sys, re, glob, json, html
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize, special, stats

RAW = Path(sys.argv[1]).expanduser()
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261019)
NBOOT = 150
LAST_YEAR = 2026            # last complete season-year (Oct 2025 - Sep 2026 water year)
FC_YEAR = 2027              # season-year being projected (Dec 2026 - May 2027)

SMAP = dict(DJF=1, JFM=2, FMA=3, MAM=4, AMJ=5, MJJ=6, JJA=7, JAS=8, ASO=9, SON=10, OND=11, NDJ=12)
MON3 = {v: k for k, v in SMAP.items()}

# Season-year Y is labelled by the calendar year in which the season ends (Dec 1997-May 1998 -> 1998).
SEASONS = {
    "decmay": dict(label="Winter + spring", months_label="Dec–May",
                   months=[(-1, 12), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
    "djf": dict(label="Winter", months_label="Dec–Feb", months=[(-1, 12), (0, 1), (0, 2)]),
    "mam": dict(label="Spring", months_label="Mar–May", months=[(0, 3), (0, 4), (0, 5)]),
    "wy": dict(label="Water year", months_label="Oct–Sep",
               months=[(-1, 10), (-1, 11), (-1, 12)] + [(0, m) for m in range(1, 10)]),
}
PERIODS = {"modern": "1951–2026", "long": "full record"}

CATS = [  # NDJ RONI bins (deg C)
    ("lnS", "Moderate–strong La Niña", -99, -1.0),
    ("lnW", "Weak La Niña", -1.0, -0.5),
    ("neu", "Neutral", -0.5, 0.5),
    ("enWM", "Weak–moderate El Niño", 0.5, 1.5),
    ("enS", "Strong El Niño", 1.5, 2.0),
    ("enVS", "Very strong El Niño", 2.0, 99),
]


def category(x):
    if x is None or not np.isfinite(x):
        return None
    if x <= -1.0:
        return "lnS"
    if x <= -0.5:
        return "lnW"
    if x < 0.5:
        return "neu"
    if x < 1.5:
        return "enWM"
    if x < 2.0:
        return "enS"
    return "enVS"


def r(v, nd=2):
    """Round for JSON; NaN -> None."""
    if v is None:
        return None
    try:
        if not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), nd)


def rl(a, nd=2):
    return [r(v, nd) for v in a]


# --------------------------------------------------------------------------------------------
# 1. ENSO: official RONI (1950-) extended back with an ERSSTv5 reconstruction of the same index
# --------------------------------------------------------------------------------------------
print("ENSO indices ...")
ds = xr.open_dataset(RAW / "ersst/sst.mnmean.nc")
sst = ds.sst
coslat = np.cos(np.deg2rad(sst.lat))


def box(lat0, lat1, lon0, lon1):
    s = sst.sel(lat=slice(lat1, lat0), lon=slice(lon0, lon1))
    return s.weighted(coslat.sel(lat=s.lat)).mean(("lat", "lon")).to_series()


def anom(s, y0=1991, y1=2020):
    b = s[(s.index.year >= y0) & (s.index.year <= y1)]
    return s - b.groupby(b.index.month).mean().reindex(s.index.month).values


n34a = anom(box(-5, 5, 190, 240))      # traditional Nino-3.4 anomaly (1991-2020 base)
tropa = anom(box(-20, 20, 0, 360))     # tropical-mean (20S-20N) anomaly
rel = n34a - tropa
base = (rel.index.year >= 1991) & (rel.index.year <= 2020)
rel_scale = n34a[base].std() / rel[base].std()
rel_monthly = rel * rel_scale
roni_v5 = rel_monthly.rolling(3, center=True).mean()      # indexed by centre month

off = pd.read_csv(RAW / "enso_indices/RONI.ascii.txt", sep=r"\s+")
off["date"] = pd.to_datetime(dict(year=off.YR, month=off.SEAS.map(SMAP), day=1))
roni_off = off.set_index("date").ANOM.sort_index()
ov = pd.concat([roni_off.rename("o"), roni_v5.rename("m")], axis=1, sort=True).dropna().loc[:"2024-12"]
cal = np.polyfit(ov.m, ov.o, 1)
recon_stats = dict(r=float(np.corrcoef(ov.m, ov.o)[0, 1]),
                   rmse=float(np.sqrt(((np.polyval(cal, ov.m) - ov.o) ** 2).mean())),
                   slope=float(cal[0]), intercept=float(cal[1]), n=int(len(ov)))
print("  reconstruction vs official RONI:", {k: round(v, 3) for k, v in recon_stats.items()})
roni = pd.Series(np.polyval(cal, roni_v5), index=roni_v5.index)
roni.loc[roni_off.index] = roni_off.values
roni_is_official = pd.Series(False, index=roni.index)
roni_is_official.loc[roni_off.index] = True

oni = pd.read_csv(RAW / "enso_indices/oni.ascii.txt", sep=r"\s+")
oni["date"] = pd.to_datetime(dict(year=oni.YR, month=oni.SEAS.map(SMAP), day=1))
oni_s = oni.set_index("date").ANOM.sort_index()


def ndj_by_season_year(s):
    d = s[s.index.month == 12]
    return pd.Series(d.values, index=d.index.year + 1)


ndj = ndj_by_season_year(roni)
ndj_off = ndj_by_season_year(roni_is_official.astype(float)).astype(bool)
oni_ndj = ndj_by_season_year(oni_s)

YEARS = np.arange(1872, LAST_YEAR + 1)
X = ndj.reindex(YEARS)

# weekly OISST Nino indices (traditional, 1991-2020 base)
wk = []
for line in open(RAW / "enso_indices/wksst9120.for"):
    m = re.match(r"\s*(\d{2}[A-Z]{3}\d{4})\s+([-\d.\s]+)$", line)
    if m:
        v = [float(t) for t in re.findall(r"-?\d+\.\d", m.group(2))]   # fixed-width: "23.4-0.4"
        if len(v) == 8:
            wk.append((pd.to_datetime(m.group(1), format="%d%b%Y"), *v))
wk = pd.DataFrame(wk, columns=["date", "n12", "n12a", "n3", "n3a", "n34", "n34a", "n4", "n4a"]).set_index("date")
wk_last = wk.iloc[-1]

pdo = pd.read_csv(RAW / "enso_indices/ersst.v5.pdo.dat", sep=r"\s+", skiprows=1, na_values=[99.99, -99.99])
pdo_long = pdo.set_index("Year").stack().dropna()
pdo_last_year = int(pdo_long.index[-1][0])
pdo_last_mon = pdo_long.index[-1][1]
pdo_last = float(pdo_long.iloc[-1])

last_month = n34a.index[-1]
current = dict(
    weekly_date=wk_last.name.strftime("%Y-%m-%d"),
    weekly_n34=float(wk_last.n34a), weekly_n3=float(wk_last.n3a), weekly_n12=float(wk_last.n12a),
    weekly_n4=float(wk_last.n4a),
    weekly_rel_cpc=2.4,  # CPC 90-day prognostic discussion, 15 Oct 2026 (relative weekly Nino-3.4)
    monthly_month=last_month.strftime("%Y-%m"),
    monthly_n34_trad=float(n34a.iloc[-1]), monthly_n34_rel=float(rel_monthly.iloc[-1]),
    roni_latest=float(roni_off.iloc[-1]), roni_latest_season=MON3[roni_off.index[-1].month] + " " + str(roni_off.index[-1].year),
    oni_latest=float(oni_s.iloc[-1]), oni_latest_season=MON3[oni_s.index[-1].month] + " " + str(oni_s.index[-1].year),
    pdo_latest=pdo_last, pdo_latest_month=f"{pdo_last_mon} {pdo_last_year}",
)
print("  current:", current)

# --------------------------------------------------------------------------------------------
# 2. CPC official forecast tables (October 2026)
# --------------------------------------------------------------------------------------------
print("CPC forecast tables ...")


def html_rows(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    for row in re.findall(r"(?is)<tr.*?</tr>", t):
        cells = [html.unescape(re.sub(r"(?s)<.*?>", "", c)).strip()
                 for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row)]
        if cells and re.match(r"^(DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\b", cells[0]):
            rows.append([cells[0][:3]] + [float(c) for c in cells[1:]])
    return rows


fdir = RAW / "forecasts"
outlook = html_rows(glob.glob(str(fdir / "*roni_outlook*.html"))[0])
strengths = html_rows(glob.glob(str(fdir / "*roni_strengths*.html"))[0])
probs = html_rows(glob.glob(str(fdir / "*roni_probabilities*.html"))[0])
# seasons start at SON 2026 (centre month Oct 2026)
fc_dates = pd.date_range("2026-10-01", periods=len(outlook), freq="MS")
fc_table = []
for d, row in zip(fc_dates, outlook):
    fc_table.append(dict(season=row[0], center=d.strftime("%Y-%m"),
                         label=f"{row[0]} {d.year if row[0] != 'NDJ' else '2026–27'}",
                         q05=row[1], q15=row[2], q25=row[3], q50=row[4], q75=row[5], q85=row[6], q95=row[7]))
ndj_fc = [f for f in fc_table if f["season"] == "NDJ"][0]
sig = np.mean([(ndj_fc["q95"] - ndj_fc["q05"]) / (2 * 1.6449),
               (ndj_fc["q85"] - ndj_fc["q15"]) / (2 * 1.0364),
               (ndj_fc["q75"] - ndj_fc["q25"]) / (2 * 0.6745)])
forecast = dict(
    issued="2026-10-08", source="NOAA CPC official RONI outlook and strength probabilities (Oct 2026)",
    ndj_median=ndj_fc["q50"], ndj_sigma=round(float(sig), 2), ndj_q05=ndj_fc["q05"], ndj_q95=ndj_fc["q95"],
    table=fc_table,
    strengths=[dict(season=s[0], p=s[1:]) for s in strengths],
    probabilities=[dict(season=s[0], lanina=s[1], neutral=s[2], elnino=s[3]) for s in probs],
    historic=dict(SON=54, OND=83, NDJ=70),  # P(3-month RONI >= +2.5), CPC ENSO discussion 8 Oct 2026
    nmme_rel_peak=3.5, cpc_consolidation_rel_peak=3.0,  # CPC 90-day discussion, 15 Oct 2026
    iri_trad_mean_ndj=3.5,  # IRI ENSO plume multi-model mean (traditional Nino-3.4), 19 Oct 2026
)
print("  NDJ forecast median/sigma:", ndj_fc["q50"], round(sig, 3))

# --------------------------------------------------------------------------------------------
# 3. Precipitation: nClimDiv divisions and GHCN-Daily stations
# --------------------------------------------------------------------------------------------
print("Precipitation ...")
cd_file = sorted(glob.glob(str(RAW / "nclimdiv/climdiv-pcpndv-v1.0.0-*")))[-1]
cd_rows = []
for line in open(cd_file):
    code = line[:10]
    st, dv, yr = int(code[:2]), int(code[2:4]), int(code[6:10])
    if st > 48:
        continue
    cd_rows.append((st * 100 + dv, yr, *[float(v) for v in line[10:].split()]))
cd = pd.DataFrame(cd_rows, columns=["div", "yr"] + list(range(1, 13))).replace(-9.99, np.nan)
cd_version = Path(cd_file).name


def season_totals(wide, months, years):
    """wide: DataFrame index=calendar year, columns 1..12 (NaN = missing). Returns Series over season-years."""
    out = np.full(len(years), np.nan)
    for i, Y in enumerate(years):
        tot = 0.0
        for oy, m in months:
            yy = Y + oy
            if yy not in wide.index:
                tot = np.nan
                break
            v = wide.at[yy, m]
            if not np.isfinite(v):
                tot = np.nan
                break
            tot += v
        out[i] = tot
    return pd.Series(out, index=years)


stations_meta = {}
for line in open(RAW / "ghcnd/ghcnd-stations.txt"):
    stations_meta[line[:11]] = dict(lat=float(line[12:20]), lon=float(line[21:30]), elev=float(line[31:37]))

PLACES = [
    dict(id="coast", kind="region", code=406, name="South Coast region",
         sub="NOAA climate division CA‑6 (coast, valleys and mountains from Santa Barbara to San Diego)"),
    dict(id="desert", kind="region", code=407, name="Southeast deserts",
         sub="NOAA climate division CA‑7 (Mojave and Colorado deserts, Coachella and Imperial valleys)"),
    dict(id="la", kind="station", code="USW00093134", name="Los Angeles", sub="Downtown (USC campus)"),
    dict(id="sd", kind="station", code="USW00023188", name="San Diego", sub="Lindbergh Field (airport)"),
    dict(id="sb", kind="station", code="USC00047902", name="Santa Barbara", sub="Downtown co-op station (gaps filled from the airport)",
         fill="USW00023190"),
    dict(id="ojai", kind="station", code="USC00046399", name="Ojai", sub="Ventura County"),
    dict(id="pas", kind="station", code="USC00046719", name="Pasadena", sub="San Gabriel Valley"),
    dict(id="sa", kind="station", code="USC00047888", name="Santa Ana", sub="Orange County (fire station)"),
    dict(id="riv", kind="station", code="USC00047470", name="Riverside", sub="Inland Empire (fire station 3)"),
    dict(id="cuy", kind="station", code="USC00042239", name="Cuyamaca", sub="San Diego County mountains, 4,650 ft"),
    dict(id="palm", kind="station", code="USC00046624", name="Palmdale", sub="Antelope Valley high desert"),
    dict(id="ps", kind="station", code="USC00046635", name="Palm Springs", sub="Coachella Valley desert"),
]

def read_station(code):
    """GHCN-Daily PRCP (inches, QC-flagged values removed) -> (daily Series, monthly wide table).
    A month counts only if at most 3 days are missing."""
    f = RAW / f"ghcnd/by_station/{code}.csv.gz"
    d = pd.read_csv(f, header=None, usecols=range(8), names=["id", "date", "el", "val", "mf", "qf", "sf", "t"],
                    dtype={"date": str, "mf": str, "qf": str, "sf": str, "t": str})
    d = d[(d.el == "PRCP") & d.qf.isna()]
    s = pd.Series(d.val.values / 254.0, index=pd.to_datetime(d.date, format="%Y%m%d"))
    s = s[~s.index.duplicated()].sort_index()
    s = s.reindex(pd.date_range(f"{s.index.min().year}-01-01", s.index.max(), freq="D"))
    ss = s[:f"{LAST_YEAR}-09-30"]
    g = ss.groupby([ss.index.year, ss.index.month])
    tot = g.sum(min_count=1)
    tot[g.apply(lambda v: v.isna().sum()) > 3] = np.nan
    return s, tot.unstack().reindex(columns=range(1, 13))


monthly_wide = {}     # place -> DataFrame (calendar year x 12)
daily = {}            # station place -> daily Series (inches, NaN missing)
so_far = {}
fill_notes = {}
for p in PLACES:
    if p["kind"] == "region":
        w = cd[cd["div"] == p["code"]].set_index("yr")[list(range(1, 13))]
        monthly_wide[p["id"]] = w
        p.update(lat=None, lon=None, elev=None, source=f"nClimDiv {p['code']:04d}")
        continue
    s, w = read_station(p["code"])
    if p.get("fill"):
        # fill missing months from a nearby station, scaled by the ratio of wet-season (Oct-May) totals
        _, w2 = read_station(p["fill"])
        w2 = w2.reindex(index=w.index.union(w2.index))
        w = w.reindex(index=w2.index)
        wet = [10, 11, 12, 1, 2, 3, 4, 5]
        both = w[wet].notna() & w2[wet].notna()
        ratio = float(w[wet].where(both).sum().sum() / w2[wet].where(both).sum().sum())
        gaps = w.isna() & w2.notna()
        w = w.where(~gaps, w2 * ratio)
        fill_notes[p["id"]] = dict(station=p["fill"], ratio=round(ratio, 3), months=int(gaps.values.sum()))
        print(f"  {p['id']}: filled {int(gaps.values.sum())} months from {p['fill']} (ratio {ratio:.3f})")
    end = s.index.max()
    daily[p["id"]] = s
    wy_start = pd.Timestamp(f"{FC_YEAR - 1}-10-01")
    tail = s[wy_start:]
    so_far[p["id"]] = dict(total=r(tail.sum(), 2), through=end.strftime("%Y-%m-%d"), missing=int(tail.isna().sum()))
    monthly_wide[p["id"]] = w
    meta = stations_meta[p["code"]]
    p.update(lat=meta["lat"], lon=meta["lon"], elev=round(meta["elev"] * 3.28084), source=f"GHCN-Daily {p['code']}")

precip = {}   # place -> season -> Series over YEARS
normals = {}
for p in PLACES:
    precip[p["id"]], normals[p["id"]] = {}, {}
    for sk, sd in SEASONS.items():
        tot = season_totals(monthly_wide[p["id"]], sd["months"], YEARS)
        precip[p["id"]][sk] = tot
        nb = tot.loc[1991:2020].dropna()
        normals[p["id"]][sk] = float(nb.mean()) if len(nb) >= 20 else float(tot.dropna().mean())
    v = precip[p["id"]]["decmay"].dropna()
    p["first"] = int(v.index.min())
    p["nseasons"] = int(len(v))
    strong_missing = [int(Y) for Y in YEARS if Y >= 1951 and X.get(Y, np.nan) >= 1.5 and not np.isfinite(precip[p["id"]]["decmay"].get(Y, np.nan))]
    print(f"  {p['id']:6s} first={p['first']} n(DecMay)={p['nseasons']} normal={normals[p['id']]['decmay']:.2f}  strong-EN seasons missing: {strong_missing}")

# --------------------------------------------------------------------------------------------
# 4. Statistical model: gamma regression with a log link
# --------------------------------------------------------------------------------------------
print("Fitting models ...")
# Mean:   log mu = b0 + bLN * min(x,0) + bEN * max(x,0)          (separate La Nina / El Nino slopes)
# Spread: shape k = exp(c0 + c1 * [x >= 0.5])                      (El Nino winters get their own spread)
# Parameter vector p = [b0, bLN, bEN, c0, c1]


def mean_design(x):
    x = np.asarray(x, float)
    return np.c_[np.ones_like(x), np.minimum(x, 0), np.maximum(x, 0)]


def shape_design(x):
    x = np.asarray(x, float)
    return np.c_[np.ones_like(x), (x >= 0.5).astype(float)]


def gamma_ll(y, ly, eta, lk):
    k = np.exp(lk)
    return k * lk - k * eta + (k - 1) * ly - k * y * np.exp(-eta) - special.gammaln(k)


def nll(p, Xm, Xk, y, ly):
    return -gamma_ll(y, ly, Xm @ p[:3], Xk @ p[3:]).sum()


def fit_model(x, y, p0=None):
    Xm, Xk, ly = mean_design(x), shape_design(x), np.log(y)
    if p0 is None:
        p0 = np.r_[np.linalg.lstsq(Xm, ly, rcond=None)[0], np.log(3.0), 0.0]
    res = optimize.minimize(nll, p0, args=(Xm, Xk, y, ly), method="BFGS")
    if not np.all(np.isfinite(res.x)) or np.abs(res.x[3:]).max() > 8:
        res = optimize.minimize(nll, p0, args=(Xm, Xk, y, ly), method="Nelder-Mead",
                                options=dict(maxiter=10000, xatol=1e-7, fatol=1e-9))
    return res.x


def predict(p, x):
    mu = np.exp(mean_design(x) @ p[:3])
    k = np.exp(shape_design(x) @ p[3:])
    return mu, k


def fit_clim(y):
    ly = np.log(y)
    f = lambda q: -gamma_ll(y, ly, q[0], q[1]).sum()
    return optimize.minimize(f, [ly.mean(), np.log(3.0)], method="BFGS").x


def crps_gamma(y, k, theta):
    F = stats.gamma.cdf(y, a=k, scale=theta)
    F1 = stats.gamma.cdf(y, a=k + 1, scale=theta)
    return y * (2 * F - 1) - k * theta * (2 * F1 - 1) - theta / special.beta(0.5, k)


models = {}
summary_rows = []
for p in PLACES:
    models[p["id"]] = {}
    for sk in SEASONS:
        models[p["id"]][sk] = {}
        for per in PERIODS:
            y_all = precip[p["id"]][sk]
            j = pd.concat([y_all.rename("p"), X.rename("x")], axis=1).dropna()
            if per == "modern":
                j = j.loc[1951:LAST_YEAR]
            else:
                j = j.loc[1896:LAST_YEAR] if p["kind"] == "region" else j.loc[:LAST_YEAR]
            yrs = j.index.values
            x = j.x.values
            y = np.maximum(j.p.values, 0.01)      # trace / zero seasons -> 0.01 in
            n = len(y)
            est = fit_model(x, y)
            # pairs bootstrap (resample seasons with replacement)
            boot = []
            while len(boot) < NBOOT:
                idx = RNG.integers(0, n, n)
                if np.sum(x[idx] >= 0.5) < 5:     # need enough El Nino seasons to identify slope and spread
                    continue
                boot.append(fit_model(x[idx], y[idx], p0=est))
            boot = np.array(boot)
            # leave-one-out cross-validation (model vs climatology)
            crps_m, crps_c, hind = np.zeros(n), np.zeros(n), []
            for i in range(n):
                m = np.ones(n, bool)
                m[i] = False
                pe = fit_model(x[m], y[m], p0=est)
                mu, k = predict(pe, x[i:i + 1])
                mu, k = mu[0], k[0]
                crps_m[i] = crps_gamma(y[i], k, mu / k)
                pc = fit_clim(y[m])
                kc = np.exp(pc[1])
                crps_c[i] = crps_gamma(y[i], kc, np.exp(pc[0]) / kc)
                if x[i] >= 1.0:
                    q = stats.gamma.ppf([0.1, 0.5, 0.9], a=k, scale=mu / k)
                    pit = stats.gamma.cdf(y[i], a=k, scale=mu / k)
                    hind.append([int(yrs[i]), r(x[i]), r(y[i]), r(q[0]), r(q[1]), r(q[2]), r(pit, 3)])

            def ss(mask):
                return r(1 - crps_m[mask].mean() / crps_c[mask].mean(), 3) if mask.sum() >= 3 else None
            norm = normals[p["id"]][sk]
            pv = j.p.values
            clim = dict(above=r(np.mean(pv > norm), 3), much=r(np.mean(pv >= 1.5 * norm), 3),
                        below75=r(np.mean(pv < 0.75 * norm), 3))
            models[p["id"]][sk][per] = dict(
                n=int(n), y0=int(yrs.min()), y1=int(yrs.max()), xmin=r(x.min()), xmax=r(x.max()),
                est=rl(est, 4), boot=[r(v, 3) for v in boot.ravel()],
                skill=dict(all=ss(np.ones(n, bool)), en=ss(x >= 1.0), strong=ss(x >= 1.5), ln=ss(x <= -0.5)),
                clim=clim, hind=hind,
                pit_in80=r(np.mean([(h[6] >= 0.1) and (h[6] <= 0.9) for h in hind]), 3) if hind else None,
            )
            # projection with the CPC default (Python check of the in-browser Monte Carlo)
            N = 60000
            xs = RNG.normal(forecast["ndj_median"], forecast["ndj_sigma"], N)
            bb = boot[RNG.integers(0, NBOOT, N)]
            mu = np.exp(bb[:, 0] + bb[:, 1] * np.minimum(xs, 0) + bb[:, 2] * np.maximum(xs, 0))
            kk = np.exp(bb[:, 3] + bb[:, 4] * (xs >= 0.5))
            draws = RNG.gamma(kk, mu / kk)
            qs = np.percentile(draws, [10, 25, 50, 75, 90])
            rec = float(pv.max())
            summary_rows.append(dict(place=p["id"], place_name=p["name"], season=sk, period=per, n=n,
                                     first=int(yrs.min()), last=int(yrs.max()),
                                     b0=est[0], b_lanina=est[1], b_elnino=est[2],
                                     shape_other=np.exp(est[3]), shape_elnino=np.exp(est[3] + est[4]),
                                     b_elnino_boot_p05=np.percentile(boot[:, 2], 5), b_elnino_boot_p95=np.percentile(boot[:, 2], 95),
                                     crpss_all=models[p['id']][sk][per]['skill']['all'],
                                     crpss_elnino=models[p['id']][sk][per]['skill']['en'],
                                     hindcast_cover80_elnino=models[p['id']][sk][per]['pit_in80'],
                                     normal_1991_2020_in=norm, record_in=rec,
                                     proj_p10=qs[0], proj_p25=qs[1], proj_median=qs[2], proj_p75=qs[3], proj_p90=qs[4],
                                     proj_median_pct_normal=100 * qs[2] / norm,
                                     proj_p_above_normal=np.mean(draws > norm),
                                     proj_p_ge150=np.mean(draws >= 1.5 * norm),
                                     proj_p_lt75=np.mean(draws < 0.75 * norm),
                                     proj_p_record=np.mean(draws > rec),
                                     clim_p_above_normal=clim["above"]))
    print(f"  {p['id']:6s} CRPSS(modern):", {sk: models[p['id']][sk]['modern']['skill']['all'] for sk in SEASONS})

summary = pd.DataFrame(summary_rows)
summary.round(4).to_csv(OUT / "model_projection_summary.csv", index=False)
print(summary[(summary.period == "modern") & (summary.season == "decmay")][
    ["place", "n", "b_elnino", "shape_elnino", "crpss_all", "crpss_elnino", "hindcast_cover80_elnino", "normal_1991_2020_in",
     "proj_median", "proj_median_pct_normal", "proj_p10", "proj_p90", "proj_p_above_normal", "clim_p_above_normal", "proj_p_record"]].round(2).to_string())

# --------------------------------------------------------------------------------------------
# 5. Monthly composites and daily extremes
# --------------------------------------------------------------------------------------------
print("Composites and extremes ...")
WY_MONTHS = [(-1, 10), (-1, 11), (-1, 12)] + [(0, m) for m in range(1, 10)]
monthly = {}
for p in PLACES:
    monthly[p["id"]] = {}
    w = monthly_wide[p["id"]]
    for per in PERIODS:
        y0 = 1951 if per == "modern" else (1896 if p["kind"] == "region" else 1872)
        arr = np.full((len(YEARS), 12), np.nan)
        for i, Y in enumerate(YEARS):
            for jm, (oy, m) in enumerate(WY_MONTHS):
                if (Y + oy) in w.index:
                    arr[i, jm] = w.at[Y + oy, m]
        out = {}
        sel = (YEARS >= y0) & (YEARS <= LAST_YEAR)
        cats = np.array([category(X.get(Y, np.nan)) for Y in YEARS])
        for c in [c[0] for c in CATS]:
            mk = sel & (cats == c)
            out[c] = dict(n=int(np.sum(mk & np.isfinite(arr).all(axis=1))), mean=rl(np.nanmean(arr[mk], axis=0) if mk.any() else [np.nan] * 12))
        out["all"] = dict(n=int(np.sum(sel)), mean=rl(np.nanmean(arr[sel], axis=0)))
        monthly[p["id"]][per] = out

extremes = {}
for p in PLACES:
    if p["kind"] != "station":
        continue
    s = daily[p["id"]]
    extremes[p["id"]] = {}
    for sk, sd in SEASONS.items():
        d1, mx = [], []
        for Y in YEARS:
            if not np.isfinite(precip[p["id"]][sk].get(Y, np.nan)):
                d1.append(None)
                mx.append(None)
                continue
            (oy0, m0), (oy1, m1) = sd["months"][0], sd["months"][-1]
            a = pd.Timestamp(year=Y + oy0, month=m0, day=1)
            b = pd.Timestamp(year=Y + oy1, month=m1, day=1) + pd.offsets.MonthEnd(0)
            v = s[a:b]
            if len(v) < (b - a).days + 1 or v.isna().sum() > 3 * len(sd["months"]):   # filled or gappy season
                d1.append(None)
                mx.append(None)
                continue
            d1.append(int((v >= 1.0).sum()))
            mx.append(r(v.max()))
        extremes[p["id"]][sk] = dict(d1=d1, max1=mx)

# --------------------------------------------------------------------------------------------
# 6. ENSO trajectories of past strong events, and the strongest-event table
# --------------------------------------------------------------------------------------------
EVENTS = [1957, 1965, 1972, 1982, 1991, 1997, 2009, 2015, 2023]
traj_events = []
for y0 in EVENTS:
    idx = pd.date_range(f"{y0}-01-01", f"{y0 + 1}-06-01", freq="MS")
    vals = roni.reindex(idx).values
    pk = roni[f"{y0}-07-01":f"{y0 + 1}-03-01"]
    traj_events.append(dict(y0=y0, label=f"{y0}–{str(y0 + 1)[2:]}", v=rl(vals), peak=r(pk.max()),
                            peak_season=MON3[pk.idxmax().month]))
idx = pd.date_range("2026-01-01", "2027-06-01", freq="MS")
cur_vals = roni_off.reindex(idx).values
centers = list(idx.strftime("%Y-%m"))
traj = dict(months=[MON3[d.month] for d in idx], events=traj_events,
            current=dict(label="2026–27", v=rl(cur_vals)),
            fc=[dict(i=centers.index(f["center"]), **f) for f in fc_table])

# --------------------------------------------------------------------------------------------
# 7. Climate-division map of the El Nino precipitation signal (1951-2026)
# --------------------------------------------------------------------------------------------
print("Division map ...")
import geopandas as gpd
from shapely.geometry import box as sbox, shape, mapping
from shapely.affinity import translate

g = gpd.read_file(glob.glob(str(RAW / "nclimdiv/shp/*.shp"))[0])
g = g.to_crs(5070)
g["geometry"] = g.geometry.simplify(3500, preserve_topology=True)
minx, miny, maxx, maxy = g.total_bounds
W = 960.0
S = W / (maxx - minx)
H = (maxy - miny) * S


def to_path(geom):
    parts = []
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        for ring in [poly.exterior] + list(poly.interiors):
            c = np.asarray(ring.coords)
            if len(c) < 4:
                continue
            xs = (c[:, 0] - minx) * S
            ys = (maxy - c[:, 1]) * S
            pts = [f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys)]
            parts.append("M" + "L".join(pts) + "Z")
    return "".join(parts)


strong_years = [Y for Y in range(1951, LAST_YEAR + 1) if X.get(Y, np.nan) >= 1.5]
map_divs = []
for _, row in g.iterrows():
    code = int(row["CLIMDIV"])
    w = cd[cd["div"] == code].set_index("yr")[list(range(1, 13))]
    if w.empty:
        continue
    vals = {}
    for sk, sd in SEASONS.items():
        tot = season_totals(w, sd["months"], np.arange(1951, LAST_YEAR + 1))
        norm = tot.loc[1991:2020].mean()
        xx = X.reindex(tot.index)
        comp = tot.loc[strong_years]
        rho = stats.spearmanr(xx.values, tot.values, nan_policy="omit").correlation
        vals[sk] = [r(100 * comp.mean() / norm, 0), r(np.mean(comp > norm), 2), r(rho, 2)]
    map_divs.append(dict(code=code, st=row["ST_ABBRV"], name=str(row["NAME"]).title(), d=to_path(row.geometry), v=vals))
states = g.dissolve(by="ST_ABBRV")
state_paths = "".join(to_path(geom) for geom in states.geometry)

# --------------------------------------------------------------------------------------------
# 8. SST anomaly maps (ERSSTv5, 1991-2020 base) for September of strong El Nino years
# --------------------------------------------------------------------------------------------
print("SST maps ...")
clim = sst.sel(time=slice("1991-01-01", "2020-12-01")).groupby("time.month").mean()
dom = dict(lat=slice(60, -30), lon=slice(120, 290))
lat_d = sst.sel(**dom).lat.values
lon_d = sst.sel(**dom).lon.values
sst_maps = []
for yr, lab in [(2026, "Sep 2026"), (2023, "Sep 2023"), (2015, "Sep 2015"), (1997, "Sep 1997"), (1982, "Sep 1982")]:
    t = pd.Timestamp(f"{yr}-09-01")
    a = (sst.sel(time=t) - clim.sel(month=9)).sel(**dom).values
    tm = float(tropa.loc[t])
    grid = [[None if not np.isfinite(v) else int(round(v * 10)) for v in rowv] for rowv in a]
    sst_maps.append(dict(year=yr, label=lab, trop=r(tm), n34=r(n34a.loc[t]), grid=grid,
                         ndj=r(X.get(yr + 1, np.nan)) if yr < 2026 else None))

land = json.load(open(RAW / "naturalearth/ne_110m_land.geojson"))
clipbox = sbox(120, -30, 290, 60)
coast = []
for ft in land["features"]:
    geom = shape(ft["geometry"])
    for shift in (0, 360):
        gg = translate(geom, xoff=shift).intersection(clipbox)
        if gg.is_empty:
            continue
        polys = [gg] if gg.geom_type == "Polygon" else [q for q in getattr(gg, "geoms", []) if q.geom_type == "Polygon"]
        for poly in polys:
            c = np.asarray(poly.exterior.coords)
            coast.append("M" + "L".join(f"{lo - 120:.2f},{60 - la:.2f}" for lo, la in c) + "Z")

# --------------------------------------------------------------------------------------------
# 9. Write outputs
# --------------------------------------------------------------------------------------------
print("Writing ...")
places_out = []
for p in PLACES:
    places_out.append({k: p[k] for k in ["id", "kind", "name", "sub", "lat", "lon", "elev", "first", "nseasons", "source"]})

data = dict(
    meta=dict(generated="2026-10-19", nclimdiv=cd_version, recon=recon_stats, last_year=LAST_YEAR, fc_year=FC_YEAR,
              nboot=NBOOT, ersst_last=str(last_month.date()), fills=fill_notes),
    current=current, forecast=forecast,
    cats=[dict(id=c[0], label=c[1], lo=c[2], hi=c[3]) for c in CATS],
    seasons={k: dict(label=v["label"], months=v["months_label"]) for k, v in SEASONS.items()},
    periods=PERIODS,
    places=places_out,
    years=YEARS.tolist(),
    roni=rl(X.values), roni_official=[bool(ndj_off.get(Y, False)) for Y in YEARS],
    oni=rl(oni_ndj.reindex(YEARS).values),
    precip={p["id"]: {sk: rl(precip[p["id"]][sk].values) for sk in SEASONS} for p in PLACES},
    normals={p["id"]: {sk: r(normals[p["id"]][sk]) for sk in SEASONS} for p in PLACES},
    models=models, monthly=monthly, extremes=extremes, so_far=so_far, traj=traj,
)
with open(OUT / "scal_data.js", "w") as f:
    f.write("// Derived data for index.html - generated by analysis/build_site_data.py on 2026-10-19\n")
    f.write("window.SCAL_DATA = ")
    json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    f.write(";\n")

geo = dict(
    divmap=dict(w=W, h=round(H, 1), divs=map_divs, states=state_paths, strong_years=strong_years,
                metrics=["Average % of normal in strong El Niño seasons", "Share of strong El Niño seasons wetter than normal",
                         "Rank correlation with El Niño strength (RONI)"]),
    sst=dict(lat=lat_d.tolist(), lon=lon_d.tolist(), maps=sst_maps, coast=coast, dlon=2, dlat=2),
)
with open(OUT / "geo_data.js", "w") as f:
    f.write("// Derived map data for index.html - generated by analysis/build_site_data.py on 2026-10-19\n")
    f.write("window.SCAL_GEO = ")
    json.dump(geo, f, separators=(",", ":"), ensure_ascii=False)
    f.write(";\n")

# tidy CSVs
rows = []
for i, Y in enumerate(YEARS):
    for p in PLACES:
        for sk in SEASONS:
            v = precip[p["id"]][sk].iloc[i]
            if np.isfinite(v):
                rows.append(dict(season_year=int(Y), season_label=f"{Y - 1}–{str(Y)[2:]}", place=p["id"], place_name=p["name"],
                                 season=sk, precip_in=round(v, 2), pct_of_1991_2020_normal=round(100 * v / normals[p["id"]][sk], 1),
                                 ndj_roni=r(X.iloc[i]), roni_source="CPC official" if ndj_off.get(Y, False) else "ERSSTv5 reconstruction",
                                 enso_category=category(X.iloc[i])))
pd.DataFrame(rows).to_csv(OUT / "seasonal_precip_enso.csv", index=False)
pd.DataFrame(dict(season_year=YEARS, season_label=[f"{Y - 1}–{str(Y)[2:]}" for Y in YEARS], ndj_roni=X.values,
                  source=["CPC official" if ndj_off.get(Y, False) else "ERSSTv5 reconstruction" for Y in YEARS],
                  ndj_oni_traditional=oni_ndj.reindex(YEARS).values)).round(2).to_csv(OUT / "enso_ndj_index.csv", index=False)
pd.DataFrame(fc_table).to_csv(OUT / "cpc_roni_outlook_oct2026.csv", index=False)
for fn in ["scal_data.js", "geo_data.js"]:
    print(f"  {fn}: {(OUT / fn).stat().st_size / 1e3:.0f} kB")
print("done")
