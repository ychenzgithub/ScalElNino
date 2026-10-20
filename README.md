# SoCal El Niño Rain Outlook (2026–27)

By **Yang Chen** and **Claude** (an AI model made by Anthropic).

An interactive web page on how the very strong 2026–27 El Niño could affect winter and spring
precipitation in Southern California. It combines 130 years of NOAA records with NOAA's October 2026
El Niño forecast, and lets readers adjust the forecast strength and uncertainty.

## View

Open `index.html` in a browser. Fonts and the d3 library load from public CDNs; everything else is local.
To serve it instead: `python3 -m http.server` in this folder, then visit <http://localhost:8000>.

## Files

| Path | What it is |
|---|---|
| `index.html` | Page structure and styles |
| `assets/app.js` | Charts, Monte Carlo projection and controls (d3 v7) |
| `data/scal_data.js` | Indices, seasonal precipitation, model ensembles, hindcasts, composites |
| `data/geo_data.js` | Climate-division map and SST anomaly maps |
| `data/*.csv` | The same results as tidy tables |
| `analysis/build_site_data.py` | Builds everything in `data/` from the raw downloads |

## Rebuild the data

Raw data live in `~/Downloads/ScalENSO/` (CPC ONI/RONI/weekly Niño indices, NCEI PDO, ERSSTv5 from NOAA PSL,
nClimDiv precipitation and climate-division shapefile, GHCN-Daily station files, the October 2026 CPC/IRI forecast
pages, and Natural Earth coastlines). Then:

```bash
rpy -I analysis/build_site_data.py ~/Downloads/ScalENSO data
```

## Method in brief

For each place and season, a gamma regression (log link) relates seasonal precipitation to the November–January
Relative Oceanic Niño Index, with separate El Niño and La Niña slopes and a separate spread for El Niño winters.
Projections combine the El Niño forecast uncertainty, 150 bootstrap refits and gamma weather noise. Skill and
calibration are checked with leave-one-out cross-validation. See the Methods and References sections of the page.

This is an independent statistical analysis, not an official forecast.
