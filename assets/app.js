/* SoCal El Niño Rain Outlook - interactive charts.
   Data: window.SCAL_DATA (data/scal_data.js) and window.SCAL_GEO (data/geo_data.js),
   both written by analysis/build_site_data.py. Library: d3 v7. */
(function () {
  'use strict';
  const D = window.SCAL_DATA;
  const G = window.SCAL_GEO;
  if (!D || !G || !window.d3) {
    const p = document.createElement('p');
    p.style.padding = '16px';
    p.textContent = 'The data files did not load. Keep the data and assets folders next to index.html.';
    document.body.prepend(p);
    return;
  }

  // ------------------------------------------------------------------ basics
  const $ = (id) => document.getElementById(id);
  const YEARS = D.years;
  const Y0 = YEARS[0];
  const YI = (Y) => Y - Y0;
  const NDJ = D.roni;
  const PLACE = Object.fromEntries(D.places.map((p) => [p.id, p]));
  const CAT_IDS = D.cats.map((c) => c.id);
  const CAT_LABEL = Object.fromEntries(D.cats.map((c) => [c.id, c.label]));
  const CAT_SHORT = { lnS: ['Mod.–strong', 'La Niña'], lnW: ['Weak', 'La Niña'], neu: ['Neutral', ''], enWM: ['Weak–mod.', 'El Niño'], enS: ['Strong', 'El Niño'], enVS: ['Very strong', 'El Niño'] };
  const catOf = (x) => (x == null ? null : x <= -1 ? 'lnS' : x <= -0.5 ? 'lnW' : x < 0.5 ? 'neu' : x < 1.5 ? 'enWM' : x < 2 ? 'enS' : 'enVS');
  const catVar = (c) => `var(--c-${c})`;
  const FC = D.forecast;
  const PLACE_LABEL = {
    coast: 'South Coast region (CA‑6)', desert: 'Southeast deserts (CA‑7)', la: 'Los Angeles (Downtown)', sd: 'San Diego (airport)',
    sb: 'Santa Barbara', ojai: 'Ojai', pas: 'Pasadena', sa: 'Santa Ana', riv: 'Riverside', cuy: 'Cuyamaca (mountains)',
    palm: 'Palmdale (high desert)', ps: 'Palm Springs',
  };
  const SEASON_PHRASE = { decmay: 'winter and spring (December–May)', djf: 'winter (December–February)', mam: 'spring (March–May)', wy: 'water year (October–September)' };
  const SEASON_SHORT = { decmay: 'Dec–May', djf: 'Dec–Feb', mam: 'Mar–May', wy: 'Oct–Sep' };

  const state = { place: 'coast', season: 'decmay', period: 'modern', mu: FC.ndj_median, sigma: FC.ndj_sigma, extrap: 'continue', sst: 0, sstRel: false, metric: 0 };

  const f0 = d3.format('.0f');
  const f1 = d3.format('.1f');
  const f2 = d3.format('.2f');
  const inch = (v) => (v == null || !isFinite(v) ? '–' : v >= 10 ? f0(v) : f1(v));
  const pct = (p) => (p < 0.01 ? '<1%' : p > 0.99 ? '>99%' : f0(100 * p) + '%');
  const signed = (v, f = f2) => (v >= 0 ? '+' : '−') + f(Math.abs(v));
  const sLabel = (Y) => `${Y - 1}–${String(Y).slice(2)}`;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const oneIn = (p) => (p < 0.02 ? 'fewer than 1 in 50' : p > 0.5 ? `${pct(p)} of` : `about 1 in ${Math.round(1 / p)}`);
  const placeName = (P) => (P.kind === 'region' ? `the ${P.name}` : P.name);

  // ------------------------------------------------------------------ model and Monte Carlo
  function model(place = state.place, season = state.season, period = state.period) {
    return D.models[place][season][period];
  }
  function bootSets(m) {
    if (!m._B) {
      const b = m.boot, B = [];
      for (let i = 0; i + 4 < b.length; i += 5) B.push([b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4]]);
      m._B = B;
    }
    return m._B;
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    const u = mulberry32(seed);
    let spare = null;
    function norm() {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let a, b, s;
      do { a = 2 * u() - 1; b = 2 * u() - 1; s = a * a + b * b; } while (s >= 1 || s === 0);
      const m = Math.sqrt((-2 * Math.log(s)) / s);
      spare = b * m;
      return a * m;
    }
    function gamma(k) { // Marsaglia-Tsang; unit scale
      if (k < 1) return gamma(k + 1) * Math.pow(u(), 1 / k);
      const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = norm(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const uu = u();
        if (uu < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    }
    return { u, norm, gamma };
  }
  // Draws of the season total: El Niño strength ~ N(mu, sigma), a bootstrap parameter set, then gamma weather noise.
  function simulate(m, mu, sigma, extrap, N = 20000, seed = 20261019) {
    const rng = makeRng(seed), B = bootSets(m), out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const x = mu + sigma * rng.norm();
      const xe = extrap === 'hold' ? Math.min(Math.max(x, m.xmin), m.xmax) : x;
      const p = B[Math.floor(rng.u() * B.length)];
      const mean = Math.exp(p[0] + p[1] * Math.min(xe, 0) + p[2] * Math.max(xe, 0));
      const k = Math.exp(p[3] + p[4] * (x >= 0.5 ? 1 : 0));
      out[i] = (rng.gamma(k) * mean) / k;
    }
    out.sort();
    return out;
  }
  const quant = (s, p) => { const i = (s.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return s[lo] + (s[hi] - s[lo]) * (i - lo); };
  const fracAbove = (s, t) => { let lo = 0, hi = s.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (s[mid] <= t) lo = mid + 1; else hi = mid; } return 1 - lo / s.length; };

  const simCache = new Map();
  function sims(place = state.place, season = state.season, period = state.period, extrap = state.extrap, mu = state.mu, sigma = state.sigma) {
    const key = [place, season, period, extrap, mu.toFixed(3), sigma.toFixed(3)].join('|');
    if (!simCache.has(key)) {
      if (simCache.size > 60) simCache.clear();
      simCache.set(key, simulate(model(place, season, period), mu, sigma, extrap));
    }
    return simCache.get(key);
  }
  const bandCache = new Map();
  function bands(place = state.place, season = state.season, period = state.period, extrap = state.extrap) {
    const key = [place, season, period, extrap].join('|');
    if (!bandCache.has(key)) {
      const m = model(place, season, period);
      const xs = d3.range(-2.6, 3.801, 0.1);
      bandCache.set(key, xs.map((x, i) => {
        const s = simulate(m, x, 0, extrap, 1500, 1000 + i);
        return { x, q10: quant(s, 0.1), q25: quant(s, 0.25), q50: quant(s, 0.5), q75: quant(s, 0.75), q90: quant(s, 0.9) };
      }));
    }
    return bandCache.get(key);
  }
  function series(place = state.place, season = state.season, period = state.period) {
    const P = D.precip[place][season], m = model(place, season, period), rows = [];
    for (let i = 0; i < YEARS.length; i++) {
      const Y = YEARS[i], v = P[i], x = NDJ[i];
      if (v == null || x == null || Y < m.y0 || Y > m.y1) continue;
      rows.push({ Y, x, p: v, cat: catOf(x), off: D.roni_official[i], oni: D.oni[i] });
    }
    return rows;
  }
  const normal = (place = state.place, season = state.season) => D.normals[place][season];
  const periodLabel = (m) => `${m.y0}–${m.y1}`;

  // ------------------------------------------------------------------ chart scaffolding
  function prep(id, height, label) {
    const el = $(id);
    const w = Math.max(260, Math.floor(el.clientWidth));
    el.replaceChildren();
    const svg = d3.select(el).append('svg').attr('width', w).attr('height', height).attr('viewBox', `0 0 ${w} ${height}`).attr('role', 'img');
    if (label) svg.attr('aria-label', label);
    return { el, svg, w, h: height };
  }
  function styleAxis(g, grid = false) {
    g.selectAll('.domain').style('stroke', 'var(--axis)');
    g.selectAll('.tick line').style('stroke', grid ? 'var(--rule)' : 'var(--axis)');
    g.selectAll('.tick text').style('fill', 'var(--muted)').style('font-size', '11px').style('font-family', 'var(--font-display)');
    return g;
  }
  function unitTop(y, n, unit = ' in') {
    const t = y.ticks(n), top = t[t.length - 1];
    return (d) => (d === top ? `${d}${unit}` : String(d));
  }
  function yGrid(svg, y, x0, x1, ticks) {
    svg.append('g').selectAll('line').data(ticks).join('line')
      .attr('x1', x0).attr('x2', x1).attr('y1', (d) => y(d)).attr('y2', (d) => y(d))
      .style('stroke', 'var(--rule)').style('stroke-width', 1);
  }
  function text(g, x, y, str, opts = {}) {
    const t = g.append('text').attr('x', x).attr('y', y).text(str)
      .style('font-family', 'var(--font-display)').style('font-size', (opts.size || 11) + 'px')
      .style('fill', opts.fill || 'var(--muted)').style('font-weight', opts.weight || 500);
    if (opts.anchor) t.attr('text-anchor', opts.anchor);
    if (opts.baseline) t.attr('dominant-baseline', opts.baseline);
    if (opts.halo) t.style('paint-order', 'stroke').style('stroke', 'var(--surface)').style('stroke-width', 3).style('stroke-linejoin', 'round');
    return t;
  }
  function barPath(x, base, w, top, r = 2) {
    const h = base - top;
    if (h <= 0.5) return `M${x},${base}h${w}v-0.5h${-w}Z`;
    r = Math.min(r, w / 2, h);
    return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`;
  }
  function diverging(stops, domain) {
    const interp = d3.interpolateRgbBasis(stops);
    return d3.scaleDiverging(interp).domain(domain).clamp(true);
  }
  const sstStops = () => [css('--sst-c2'), css('--sst-c1'), css('--div-mid'), css('--sst-w1'), css('--sst-w2')];
  const wetStops = () => [css('--dry-2'), css('--dry-1'), css('--div-mid'), css('--wet-1'), css('--wet-2')];

  // tooltip
  const tip = $('tip');
  function showTip(evt, title, rows) {
    tip.replaceChildren();
    if (title) { const t = document.createElement('div'); t.className = 'tip-title'; t.textContent = title; tip.append(t); }
    for (const r of rows) {
      const d = document.createElement('div'); d.className = 'tip-row';
      if (r.color) { const k = document.createElement('span'); k.className = 'tip-key'; k.style.background = r.color; d.append(k); }
      const v = document.createElement('strong'); v.textContent = r.value;
      const l = document.createElement('span'); l.textContent = r.label || '';
      d.append(v, l); tip.append(d);
    }
    tip.hidden = false;
    const pad = 14, rect = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
    tip.style.transform = `translate(${Math.max(4, x)}px, ${Math.max(4, y)}px)`;
  }
  const hideTip = () => { tip.hidden = true; };
  document.addEventListener('pointerdown', (e) => { if (!e.target.closest('.chart')) hideTip(); });
  window.addEventListener('scroll', hideTip, { passive: true });

  function legendCats(id, cats = CAT_IDS) {
    const el = $(id); el.replaceChildren();
    for (const c of cats) {
      const s = document.createElement('span'); const i = document.createElement('i');
      i.style.background = catVar(c); s.append(i, document.createTextNode(CAT_LABEL[c])); el.append(s);
    }
  }

  // ------------------------------------------------------------------ hero stripes
  function renderStripes() {
    const el = $('stripes');
    const w0 = el.clientWidth;
    const h = w0 < 560 ? 104 : 132;
    const { svg, w } = prep('stripes', h, 'El Niño–La Niña strength each winter since 1872–73, with the 2026–27 forecast as the last stripe');
    const yrs = YEARS.filter((Y) => NDJ[YI(Y)] != null && Y >= 1873);
    const top = 36, bottom = 22;
    const fcW = Math.max(12, w * 0.02), gap = Math.max(8, w * 0.012);
    const x = d3.scaleBand().domain(yrs).range([0, w - fcW - gap]);
    const color = diverging(sstStops(), [-2.6, 0, 2.6]);
    svg.append('g').selectAll('rect').data(yrs).join('rect')
      .attr('x', (Y) => x(Y)).attr('y', top).attr('width', x.bandwidth() + 0.5).attr('height', h - top - bottom)
      .style('fill', (Y) => color(NDJ[YI(Y)]))
      .on('pointermove', (e, Y) => {
        const v = NDJ[YI(Y)];
        showTip(e, sLabel(Y), [{ value: signed(v) + ' °C', label: ' ' + CAT_LABEL[catOf(v)] }, { value: '', label: Y < 1950 ? 'Reconstructed from ERSSTv5' : 'NOAA official index' }]);
      })
      .on('pointerleave', hideTip);
    const fx = w - fcW;
    svg.append('rect').attr('x', fx).attr('y', top).attr('width', fcW).attr('height', h - top - bottom).attr('rx', 2)
      .style('fill', color(state.mu)).style('stroke', 'var(--fc)').style('stroke-width', 2.5)
      .on('pointermove', (e) => showTip(e, '2026–27 forecast', [{ value: signed(state.mu) + ' °C', label: ` ± ${f2(state.sigma)} (Nov–Jan RONI)` }]))
      .on('pointerleave', hideTip);
    text(svg, w, top - 22, w < 560 ? '2026–27' : '2026–27 forecast', { anchor: 'end', fill: 'var(--fc-text)', weight: 700, size: 12 });
    svg.append('line').attr('x1', fx + fcW / 2).attr('x2', fx + fcW / 2).attr('y1', top - 18).attr('y2', top).style('stroke', 'var(--fc)').style('stroke-width', 1);
    const marks = [[1878, '1877–78'], [1983, '1982–83'], [1998, '1997–98'], [2016, '2015–16']];
    const g = svg.append('g');
    marks.forEach(([Y, lab], i) => {
      if (x(Y) == null) return;
      const cx = x(Y) + x.bandwidth() / 2;
      const ly = w < 700 && i === 2 ? top - 22 : top - 8;
      g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', ly + 3).attr('y2', top).style('stroke', 'var(--ink-2)').style('stroke-width', 1);
      if (!(w < 700 && i === 3 && fx - cx < 60)) text(g, cx, ly, w < 560 ? "'" + String(Y - 1).slice(2) : lab, { anchor: 'middle', fill: 'var(--ink-2)', size: 11 });
    });
    const ticks = [1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020].filter((t) => x(t) != null);
    ticks.forEach((t) => text(svg, x(t) + x.bandwidth() / 2, h - 6, String(t), { anchor: 'middle', size: 11 }));
    const x1950 = x(1950);
    svg.append('line').attr('x1', x1950).attr('x2', x1950).attr('y1', top).attr('y2', h - bottom + 4).style('stroke', 'var(--surface)').style('stroke-width', 1.5);
  }

  // ------------------------------------------------------------------ strength ruler
  function renderRuler() {
    const el = $('ruler');
    const narrow = el.clientWidth < 560;
    const evs = [];
    for (let i = 0; i < YEARS.length; i++) if (YEARS[i] >= 1951 && NDJ[i] >= 1.4) evs.push({ Y: YEARS[i], x: NDJ[i] });
    const { svg, w, h } = prep('ruler', narrow ? 176 : 160, 'Forecast distribution of El Niño strength with past El Niño winters marked');
    const ml = 14, mr = 14, axisY = 96;
    const x = d3.scaleLinear().domain([0.5, 3.8]).range([ml, w - mr]);
    const bandsDef = [[0.5, 1, 'Weak'], [1, 1.5, 'Moderate'], [1.5, 2, 'Strong'], [2, 2.5, narrow ? 'V. strong' : 'Very strong'], [2.5, 3.8, narrow ? 'Beyond record' : 'Beyond any since 1950']];
    bandsDef.forEach(([a, b, lab], i) => {
      svg.append('rect').attr('x', x(a)).attr('y', 16).attr('width', x(b) - x(a)).attr('height', axisY - 16)
        .style('fill', i % 2 ? 'var(--wash)' : 'transparent');
      text(svg, (x(a) + x(b)) / 2, 11, lab, { anchor: 'middle', size: 10.5, fill: 'var(--muted)' });
    });
    // forecast density
    const s = Math.max(state.sigma, 0.001);
    const xs = d3.range(0.5, 3.801, 0.01);
    const pdf = xs.map((v) => Math.exp(-0.5 * ((v - state.mu) / s) ** 2));
    const yd = (p) => axisY - 4 - p * 60;
    if (state.sigma >= 0.02) {
      svg.append('path').datum(xs.map((v, i) => [v, pdf[i]]))
        .attr('d', d3.area().x((d) => x(d[0])).y0(axisY).y1((d) => yd(d[1])).curve(d3.curveBasis))
        .style('fill', 'var(--fc-wash-2)').style('stroke', 'var(--fc)').style('stroke-width', 2);
    }
    svg.append('line').attr('x1', x(state.mu)).attr('x2', x(state.mu)).attr('y1', yd(1) - 2).attr('y2', axisY).style('stroke', 'var(--fc)').style('stroke-width', 2.5);
    text(svg, Math.min(Math.max(x(state.mu), 40), w - 40), yd(1) - 7, signed(state.mu) + ' °C', { anchor: 'middle', fill: 'var(--fc-text)', weight: 750, size: 13, halo: true });
    // axis
    const ax = svg.append('g').attr('transform', `translate(0,${axisY})`).call(d3.axisBottom(x).tickValues([0.5, 1, 1.5, 2, 2.5, 3, 3.5]).tickFormat((d) => signed(d, f1)).tickSize(4));
    styleAxis(ax);
    // NOAA percentile marks and past events, labels placed greedily in rows
    const items = [
      { x: FC.ndj_q05, lab: 'NOAA 5%', noaa: true }, { x: FC.ndj_q95, lab: 'NOAA 95%', noaa: true },
      ...evs.map((e) => ({ x: e.x, lab: sLabel(e.Y), Y: e.Y })),
    ].sort((a, b) => a.x - b.x);
    const rows = [];
    const rowY = (r) => axisY + 32 + r * 14;
    items.forEach((it) => {
      const px = x(it.x), wlab = it.lab.length * 6.2 + 8;
      let r = 0;
      while (rows[r] != null && rows[r] > px - wlab / 2) r++;
      rows[r] = px + wlab / 2;
      it.r = r; it.px = px;
    });
    const g = svg.append('g');
    items.forEach((it) => {
      const col = it.noaa ? 'var(--fc)' : catVar(catOf(it.x));
      g.append('line').attr('x1', it.px).attr('x2', it.px).attr('y1', axisY - 10).attr('y2', rowY(it.r) - 10)
        .style('stroke', it.noaa ? 'var(--fc)' : 'var(--ink-2)').style('stroke-width', 1).style('opacity', 0.6);
      g.append('circle').attr('cx', it.px).attr('cy', axisY - 10).attr('r', 4.5).style('fill', col).style('stroke', 'var(--surface)').style('stroke-width', 1.5);
      const t = text(g, it.px, rowY(it.r), it.lab, { anchor: 'middle', size: 11, fill: it.noaa ? 'var(--fc-text)' : 'var(--ink-2)', weight: it.noaa ? 650 : 550 });
      const hit = g.append('rect').attr('x', it.px - 24).attr('y', axisY - 22).attr('width', 48).attr('height', rowY(it.r) - axisY + 26)
        .style('fill', 'transparent').style('cursor', 'pointer').attr('tabindex', 0).attr('role', 'button')
        .attr('aria-label', `Set El Niño strength to ${signed(it.x)} °C (${it.lab})`);
      const choose = () => setMu(it.x);
      hit.on('click', choose).on('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } })
        .on('pointermove', (e) => showTip(e, it.noaa ? it.lab : sLabel(it.Y), [{ value: signed(it.x) + ' °C', label: it.noaa ? ' NOAA forecast percentile' : ' ' + CAT_LABEL[catOf(it.x)] }, { value: '', label: 'Select to use this strength' }]))
        .on('pointerleave', hideTip);
      void t;
    });
    svg.attr('height', Math.max(h, rowY(rows.length - 1) + 8)).attr('viewBox', `0 0 ${w} ${Math.max(h, rowY(rows.length - 1) + 8)}`);
  }

  function renderPresets() {
    const el = $('presets'); el.replaceChildren();
    const ps = [
      ['NOAA median', FC.ndj_median], ['NOAA low (5%)', FC.ndj_q05], ['NOAA high (95%)', FC.ndj_q95],
      ['Model peak (NMME)', FC.nmme_rel_peak], ['Like 1997–98', NDJ[YI(1998)]], ['Like 2015–16', NDJ[YI(2016)]], ['Moderate El Niño', 1.25],
    ];
    for (const [lab, v] of ps) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip';
      b.textContent = `${lab} ${signed(v)}`; b.addEventListener('click', () => setMu(v)); el.append(b);
    }
  }

  // ------------------------------------------------------------------ outlook
  function outlookStats() {
    const m = model(), s = sims(), norm = normal(), rows = series();
    let rec = rows[0];
    rows.forEach((r) => { if (r.p > rec.p) rec = r; });
    return {
      m, s, norm, rows, rec,
      q10: quant(s, 0.1), q25: quant(s, 0.25), med: quant(s, 0.5), q75: quant(s, 0.75), q90: quant(s, 0.9),
      pA: fracAbove(s, norm), pM: fracAbove(s, 1.5 * norm), pB: 1 - fracAbove(s, 0.75 * norm), pR: fracAbove(s, rec.p),
    };
  }

  function renderTiles(st) {
    const P = PLACE[state.place];
    $('t-proj-label').textContent = `Projected rain, ${P.name}, ${SEASON_SHORT[state.season]}`;
    $('t-proj').textContent = `${inch(st.med)} in`;
    $('t-proj-note').textContent = `Middle estimate, ${f0((100 * st.med) / st.norm)}% of normal. ${pct(st.pA)} chance of above normal, versus ${pct(st.m.clim.above)} in a typical year.`;
    $('t-weekly').textContent = signed(D.current.weekly_n34, f1) + ' °C';
  }

  function renderOutlook(st) {
    const P = PLACE[state.place];
    const { m, norm, rows, rec } = st;
    const lean = st.pA >= 0.75 ? 'strongly favors' : st.pA >= 0.6 ? 'favors' : st.pA >= 0.45 ? 'slightly favors' : 'does not favor';
    const r16 = rows.find((r) => r.Y === 2016);
    const dud = r16 && r16.p < 0.75 * norm ? `, as 2015–16 did (${f0((100 * r16.p) / norm)}% of normal)` : '';
    $('verdict').textContent =
      `With El Niño at ${signed(state.mu)} °C (± ${f2(state.sigma)}), history ${lean} a wetter-than-normal ${SEASON_PHRASE[state.season]} in ${placeName(P)}. ` +
      `The middle estimate is ${inch(st.med)} inches, ${f0((100 * st.med) / norm)}% of the 1991–2020 normal of ${inch(norm)} inches. ` +
      `The chance of above-normal precipitation is ${pct(st.pA)}, compared with ${pct(m.clim.above)} in a typical year. ` +
      `Still, ${oneIn(st.pB)} seasons like this would come in below 75% of normal${dud}.`;
    $('b-med').innerHTML = '';
    $('b-med').append(document.createTextNode(`${inch(st.med)} in `));
    const sm = document.createElement('small'); sm.textContent = `${f0((100 * st.med) / norm)}% of normal`; $('b-med').append(sm);
    $('b-med-s').textContent = `Normal (1991–2020): ${inch(norm)} in`;
    $('b-iqr').textContent = `${inch(st.q25)}–${inch(st.q75)} in`;
    $('b-iqr-s').textContent = `${f0((100 * st.q25) / norm)}–${f0((100 * st.q75) / norm)}% of normal`;
    $('b-80').textContent = `${inch(st.q10)}–${inch(st.q90)} in`;
    $('b-80-s').textContent = `${f0((100 * st.q10) / norm)}–${f0((100 * st.q90) / norm)}% of normal`;

    // meters
    const n = rows.length;
    const meters = [
      { lab: `Above normal (more than ${inch(norm)} in)`, p: st.pA, c: m.clim.above },
      { lab: `Much above normal (150% or more, ${inch(1.5 * norm)} in)`, p: st.pM, c: m.clim.much },
      { lab: `Well below normal (under 75%, ${inch(0.75 * norm)} in)`, p: st.pB, c: m.clim.below75 },
      { lab: `Wettest since ${m.y0} (beat ${inch(rec.p)} in, ${sLabel(rec.Y)})`, p: st.pR, c: 1 / n },
    ];
    const box = $('meters'); box.replaceChildren();
    for (const mt of meters) {
      const row = document.createElement('div'); row.className = 'meter';
      const l = document.createElement('div'); l.className = 'ml'; l.textContent = mt.lab;
      const tr = document.createElement('div'); tr.className = 'track';
      tr.setAttribute('role', 'img'); tr.setAttribute('aria-label', `${pct(mt.p)} projected, ${pct(mt.c)} in a typical year`);
      const fl = document.createElement('div'); fl.className = 'fill'; fl.style.width = (100 * mt.p).toFixed(1) + '%';
      const cl = document.createElement('div'); cl.className = 'clim'; cl.style.left = `calc(${(100 * mt.c).toFixed(1)}% - 1px)`;
      tr.append(fl, cl);
      const v = document.createElement('div'); v.className = 'mv'; v.textContent = pct(mt.p);
      const note = document.createElement('div'); note.className = 'mn'; note.textContent = `Typical year: ${pct(mt.c)}`;
      row.append(l, tr, v, note); box.append(row);
    }
    const sf = D.so_far[state.place];
    const sfEl = $('so-far');
    if (P.kind === 'station' && sf && sf.through >= '2026-10-10' && sf.total != null) {
      sfEl.hidden = false;
      sfEl.textContent = `So far this water year (since October 1): ${f2(sf.total)} in at ${P.name}, through ${d3.timeFormat('%B %-d')(new Date(sf.through + 'T12:00:00'))}.`;
    } else sfEl.hidden = true;

    renderRanked(st);
    renderDensity(st);
    renderSensitivity(st);
  }

  function renderRanked(st) {
    const P = PLACE[state.place];
    const rows = st.rows.slice().sort((a, b) => a.p - b.p);
    const el = $('ranked');
    const H = el.clientWidth < 520 ? 260 : 320;
    const { svg, w, h } = prep('ranked', H, `Past seasons sorted from driest to wettest with the 2026–27 projected range`);
    const ml = 44, mr = 10, mt = 12, mb = 26;
    const ymax = Math.max(rows[rows.length - 1].p, st.q90) * 1.06;
    const x = d3.scaleBand().domain(rows.map((r) => r.Y)).range([ml, w - mr]).paddingInner(0.18);
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([h - mb, mt]);
    yGrid(svg, y, ml, w - mr, y.ticks(5));
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(unitTop(y, 5)).tickSize(0).tickPadding(6))).select('.domain').remove();
    // projection bands
    svg.append('rect').attr('x', ml).attr('width', w - ml - mr).attr('y', y(st.q90)).attr('height', y(st.q10) - y(st.q90)).style('fill', 'var(--fc-wash)');
    svg.append('rect').attr('x', ml).attr('width', w - ml - mr).attr('y', y(st.q75)).attr('height', y(st.q25) - y(st.q75)).style('fill', 'var(--fc-wash-2)');
    const bw = Math.min(24, x.bandwidth()), off = (x.bandwidth() - bw) / 2;
    svg.append('g').selectAll('path').data(rows).join('path')
      .attr('d', (r) => barPath(x(r.Y) + off, y(0), bw, y(r.p), Math.min(3, bw / 2)))
      .style('fill', (r) => catVar(r.cat))
      .on('pointermove', (e, r) => showTip(e, sLabel(r.Y), [
        { value: `${f1(r.p)} in`, label: ` ${f0((100 * r.p) / st.norm)}% of normal` },
        { value: signed(r.x) + ' °C', label: ' ' + CAT_LABEL[r.cat], color: catVar(r.cat) }]))
      .on('pointerleave', hideTip);
    // normal line and median
    svg.append('line').attr('x1', ml).attr('x2', w - mr).attr('y1', y(st.norm)).attr('y2', y(st.norm)).style('stroke', 'var(--ink-2)').style('stroke-width', 1);
    svg.append('line').attr('x1', ml).attr('x2', w - mr).attr('y1', y(st.med)).attr('y2', y(st.med)).style('stroke', 'var(--fc)').style('stroke-width', 2.5);
    const below = rows.filter((r) => r.p < st.med).length;
    const rank = rows.length - below + 1;
    const xr = below < rows.length ? x(rows[below].Y) - (x.step() * x.paddingInner()) / 2 : w - mr;
    svg.append('circle').attr('cx', xr).attr('cy', y(st.med)).attr('r', 5).style('fill', 'var(--fc)').style('stroke', 'var(--surface)').style('stroke-width', 2);
    const lowSide = st.norm < st.med;
    text(svg, ml + 6, y(st.med) - 7, `2026–27 middle estimate ${inch(st.med)} in`, { fill: 'var(--fc-text)', weight: 700, size: 12, halo: true });
    text(svg, ml + 6, lowSide ? y(st.norm) + 14 : y(st.norm) - 6, `Normal ${inch(st.norm)} in`, { fill: 'var(--ink-2)', size: 11, halo: true });
    text(svg, ml, h - 8, 'Driest', { size: 11 });
    text(svg, w - mr, h - 8, 'Wettest', { anchor: 'end', size: 11 });
    $('rank-sub').textContent = `${P.name}, ${SEASON_SHORT[state.season]}: every season ${periodLabel(st.m)}, driest to wettest, colored by that winter's El Niño–La Niña state. The middle estimate would rank ${ordinal(rank)} wettest of ${rows.length + 1}. Dark teal band: middle 50%; light band: 8 in 10.`;
  }
  const ordinal = (n) => { const s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };

  function kde(values, grid, bw, weights) {
    const out = new Float64Array(grid.length);
    const c = 1 / (bw * Math.sqrt(2 * Math.PI));
    for (let j = 0; j < values.length; j++) {
      const v = values[j], wj = weights ? weights[j] : 1;
      for (let i = 0; i < grid.length; i++) {
        const z1 = (grid[i] - v) / bw, z2 = (grid[i] + v) / bw; // reflect at zero
        out[i] += wj * c * (Math.exp(-0.5 * z1 * z1) + Math.exp(-0.5 * z2 * z2));
      }
    }
    const tot = weights ? weights.reduce((a, b) => a + b, 0) : values.length;
    return Array.from(out, (v) => v / tot);
  }
  function renderDensity(st) {
    const { svg, w, h } = prep('density', 230, 'Probability of each seasonal total for the 2026–27 projection and for all past seasons');
    const ml = 12, mr = 12, mt = 10, mb = 30;
    const vals = st.rows.map((r) => r.p);
    const xmax = Math.max(quant(st.s, 0.985), d3.max(vals) * 1.05);
    const x = d3.scaleLinear().domain([0, xmax]).nice().range([ml, w - mr]);
    const grid = d3.range(0, x.domain()[1], x.domain()[1] / 160);
    const sd = d3.deviation(vals), bwH = 1.06 * sd * Math.pow(vals.length, -0.2);
    const dH = kde(vals, grid, bwH);
    const sub = []; for (let i = 0; i < st.s.length; i += 8) sub.push(st.s[i]);
    const bwP = 1.06 * d3.deviation(sub) * Math.pow(sub.length, -0.2);
    const dP = kde(sub, grid, bwP);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(dH), d3.max(dP)) * 1.08]).range([h - mb, mt]);
    const area = d3.area().x((d, i) => x(grid[i])).y0(y(0)).y1((d) => y(d)).curve(d3.curveBasis);
    const line = d3.line().x((d, i) => x(grid[i])).y((d) => y(d)).curve(d3.curveBasis);
    svg.append('path').datum(dH).attr('d', area).style('fill', 'var(--wash)');
    svg.append('path').datum(dH).attr('d', line).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 1.5);
    svg.append('path').datum(dP).attr('d', area).style('fill', 'var(--fc-wash-2)');
    svg.append('path').datum(dP).attr('d', line).style('fill', 'none').style('stroke', 'var(--fc)').style('stroke-width', 2);
    svg.append('line').attr('x1', x(st.norm)).attr('x2', x(st.norm)).attr('y1', mt).attr('y2', y(0)).style('stroke', 'var(--ink-2)').style('stroke-width', 1);
    text(svg, x(st.norm) + 4, mt + 10, 'Normal', { size: 11, fill: 'var(--ink-2)', halo: true });
    styleAxis(svg.append('g').attr('transform', `translate(0,${y(0)})`).call(d3.axisBottom(x).ticks(6).tickSize(4)));
    text(svg, w - mr, h - 2, 'inches', { anchor: 'end', size: 10.5 });
    // hover readout
    const hit = svg.append('rect').attr('x', ml).attr('y', mt).attr('width', w - ml - mr).attr('height', y(0) - mt).style('fill', 'transparent');
    const cross = svg.append('line').attr('y1', mt).attr('y2', y(0)).style('stroke', 'var(--axis)').style('opacity', 0);
    hit.on('pointermove', (e) => {
      const [px] = d3.pointer(e); const v = x.invert(px);
      cross.attr('x1', px).attr('x2', px).style('opacity', 1);
      const pP = fracAbove(st.s, v), pH = st.rows.filter((r) => r.p > v).length / st.rows.length;
      showTip(e, `More than ${inch(v)} in`, [{ value: pct(pP), label: ' 2026–27 projection', color: 'var(--fc)' }, { value: pct(pH), label: ' of past seasons', color: 'var(--ink-2)' }]);
    }).on('pointerleave', () => { hideTip(); cross.style('opacity', 0); });
  }

  function renderSensitivity(st) {
    const tbl = $('sens'); tbl.replaceChildren();
    const norm = st.norm;
    const head = tbl.createTHead().insertRow();
    ['Approach', 'Middle', '8‑in‑10 range', 'Above normal'].forEach((t, i) => { const th = document.createElement('th'); th.textContent = t; if (i) th.className = 'n'; head.append(th); });
    const body = tbl.createTBody();
    const combos = [['modern', 'continue'], ['modern', 'hold'], ['long', 'continue'], ['long', 'hold']];
    for (const [per, ex] of combos) {
      const m = model(state.place, state.season, per);
      const s = sims(state.place, state.season, per, ex);
      const row = body.insertRow();
      if (per === state.period && ex === state.extrap) row.className = 'main';
      const lab = `${per === 'modern' ? 'Since 1951' : `Full record (${m.y0}–)`}, ${ex === 'continue' ? 'trend continues' : 'no extra rain past record strength'}${per === state.period && ex === state.extrap ? ' (shown above)' : ''}`;
      const n = normal(state.place, state.season);
      [lab, `${inch(quant(s, 0.5))} in · ${f0((100 * quant(s, 0.5)) / n)}%`, `${inch(quant(s, 0.1))}–${inch(quant(s, 0.9))} in`, pct(fracAbove(s, n))]
        .forEach((t, i) => { const c = row.insertCell(); c.textContent = t; if (i) c.className = 'n'; });
    }
    const strong = st.rows.filter((r) => r.x >= 1.5).sort((a, b) => a.p - b.p);
    if (strong.length) {
      const vals = strong.map((r) => r.p);
      const k = strong.filter((r) => r.p > norm).length;
      const row = body.insertRow();
      [`History only: the ${strong.length} strong El Niño seasons, ${periodLabel(st.m)}`,
        `${inch(d3.median(vals))} in · ${f0((100 * d3.median(vals)) / norm)}%`,
        `${inch(vals[0])}–${inch(vals[vals.length - 1])} in (all)`, `${k} of ${strong.length}`]
        .forEach((t, i) => { const c = row.insertCell(); c.textContent = t; if (i) c.className = 'n'; });
    }
  }

  // ------------------------------------------------------------------ history: scatter
  function renderScatter(st) {
    const P = PLACE[state.place];
    const el = $('scatter');
    const H = el.clientWidth < 560 ? 320 : 400;
    const { svg, w, h } = prep('scatter', H, `Season precipitation versus El Niño strength at ${P.name}`);
    const ml = 46, mr = 14, mt = 22, mb = 40;
    const rows = st.rows, m = st.m;
    const bd = bands();
    const ymax = Math.max(d3.max(rows, (r) => r.p) * 1.1, st.q90 * 1.02);
    const x = d3.scaleLinear().domain([-2.6, 3.8]).range([ml, w - mr]);
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([h - mb, mt]);
    const clipId = 'clip-sc';
    svg.append('clipPath').attr('id', clipId).append('rect').attr('x', ml).attr('y', mt).attr('width', w - ml - mr).attr('height', h - mb - mt);
    yGrid(svg, y, ml, w - mr, y.ticks(6));
    // beyond-record zone
    svg.append('rect').attr('x', x(m.xmax)).attr('y', mt).attr('width', x(3.8) - x(m.xmax)).attr('height', h - mb - mt).style('fill', 'var(--wash)').style('opacity', 0.7);
    text(svg, w - mr - 4, h - mb - 6, w < 560 ? 'Beyond record' : `Stronger than any El Niño, ${m.y0}–${m.y1}`, { anchor: 'end', size: 10.5, fill: 'var(--ink-2)', halo: true });
    // forecast band
    svg.append('rect').attr('x', x(state.mu - state.sigma)).attr('width', Math.max(2, x(state.mu + state.sigma) - x(state.mu - state.sigma)))
      .attr('y', mt).attr('height', h - mb - mt).style('fill', 'var(--fc-wash)');
    svg.append('line').attr('x1', x(state.mu)).attr('x2', x(state.mu)).attr('y1', mt).attr('y2', h - mb).style('stroke', 'var(--fc)').style('stroke-width', 2);
    text(svg, x(state.mu), mt - 6, '2026–27 forecast', { anchor: 'middle', size: 11, fill: 'var(--fc-text)', weight: 700 });
    // model bands
    const gb = svg.append('g').attr('clip-path', `url(#${clipId})`);
    const ar = (lo, hi) => d3.area().x((d) => x(d.x)).y0((d) => y(d[lo])).y1((d) => y(d[hi])).curve(d3.curveMonotoneX);
    gb.append('path').datum(bd).attr('d', ar('q10', 'q90')).style('fill', 'var(--ink-2)').style('opacity', 0.09);
    gb.append('path').datum(bd).attr('d', ar('q25', 'q75')).style('fill', 'var(--ink-2)').style('opacity', 0.12);
    const inR = bd.filter((d) => d.x <= m.xmax + 1e-9), outR = bd.filter((d) => d.x >= m.xmax - 0.1);
    const ln = d3.line().x((d) => x(d.x)).y((d) => y(d.q50)).curve(d3.curveMonotoneX);
    gb.append('path').datum(inR).attr('d', ln).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 2);
    gb.append('path').datum(outR).attr('d', ln).style('fill', 'none').style('stroke', 'var(--ink-2)').style('stroke-width', 2).style('stroke-dasharray', '5 4');
    // normal
    svg.append('line').attr('x1', ml).attr('x2', w - mr).attr('y1', y(st.norm)).attr('y2', y(st.norm)).style('stroke', 'var(--ink-2)').style('stroke-width', 1).style('opacity', 0.6);
    text(svg, ml + 4, y(st.norm) - 5, `Normal ${inch(st.norm)} in`, { size: 10.5, fill: 'var(--ink-2)', halo: true });
    // axes
    styleAxis(svg.append('g').attr('transform', `translate(0,${h - mb})`).call(d3.axisBottom(x).tickValues([-2, -1, 0, 1, 2, 3]).tickFormat((d) => (d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d)).tickSize(4)));
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(unitTop(y, 6)).tickSize(0).tickPadding(6))).select('.domain').remove();
    text(svg, x(-2.5), h - 6, '← La Niña', { size: 11, fill: 'var(--c-lnS)', weight: 650 });
    text(svg, w - mr, h - 6, 'El Niño →', { anchor: 'end', size: 11, fill: 'var(--c-enS)', weight: 650 });
    text(svg, (ml + w - mr) / 2, h - 6, 'Nov–Jan relative Niño‑3.4 (°C)', { anchor: 'middle', size: 11 });
    // points
    const pts = rows.map((r) => ({ ...r, cx: x(r.x), cy: y(r.p) }));
    const gp = svg.append('g');
    gp.selectAll('circle').data(pts).join('circle')
      .attr('cx', (d) => d.cx).attr('cy', (d) => d.cy).attr('r', 4.5)
      .style('fill', (d) => (d.off ? catVar(d.cat) : 'var(--surface)'))
      .style('stroke', (d) => (d.off ? 'var(--surface)' : catVar(d.cat))).style('stroke-width', (d) => (d.off ? 1.5 : 2));
    // labels
    const wettest = pts.reduce((a, b) => (b.p > a.p ? b : a), pts[0]);
    const labYears = new Set([1983, 1998, 2016, 2024, wettest.Y]);
    const placed = [];
    pts.filter((d) => labYears.has(d.Y)).sort((a, b) => b.cy - a.cy).forEach((d) => {
      let ly = d.cy - 8;
      const lx = d.cx > w - 90 ? d.cx - 8 : d.cx + 8;
      for (const p of placed) if (Math.abs(p.x - lx) < 60 && Math.abs(p.y - ly) < 12) ly = p.y - 13;
      placed.push({ x: lx, y: ly });
      text(svg, lx, ly, sLabel(d.Y), { anchor: d.cx > w - 90 ? 'end' : 'start', size: 11, fill: 'var(--ink-2)', weight: 600, halo: true });
    });
    // hover via Delaunay
    const del = d3.Delaunay.from(pts, (d) => d.cx, (d) => d.cy);
    const ring = svg.append('circle').attr('r', 8).style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 1.5).style('opacity', 0);
    svg.append('rect').attr('x', ml).attr('y', mt).attr('width', w - ml - mr).attr('height', h - mb - mt).style('fill', 'transparent')
      .on('pointermove', (e) => {
        const [px, py] = d3.pointer(e); const i = del.find(px, py); const d = pts[i];
        if (!d || Math.hypot(d.cx - px, d.cy - py) > 28) { ring.style('opacity', 0); hideTip(); return; }
        ring.attr('cx', d.cx).attr('cy', d.cy).style('opacity', 1);
        showTip(e, sLabel(d.Y), [
          { value: `${f1(d.p)} in`, label: ` ${f0((100 * d.p) / st.norm)}% of normal` },
          { value: signed(d.x) + ' °C', label: ` ${CAT_LABEL[d.cat]}${d.off ? '' : ' (reconstructed index)'}`, color: catVar(d.cat) },
          ...(d.oni != null ? [{ value: signed(d.oni) + ' °C', label: ' traditional ONI' }] : [])]);
      })
      .on('pointerleave', () => { ring.style('opacity', 0); hideTip(); });
    const desc = P.kind === 'region' ? `NOAA climate division CA‑${P.id === 'coast' ? 6 : 7}` : P.sub;
    $('scatter-sub').textContent = `Each dot is one season ${P.kind === 'region' ? 'in' : 'at'} ${placeName(P)} (${desc}), ${SEASON_SHORT[state.season]}, ${periodLabel(m)}. The dashed line extends the model beyond the strongest El Niño in this record.`;
  }

  // ------------------------------------------------------------------ history: categories strip
  function renderCats(st) {
    const el = $('cats');
    const H = el.clientWidth < 420 ? 300 : 320;
    const { svg, w, h } = prep('cats', H, 'Season precipitation as percent of normal by El Niño–La Niña category');
    const ml = 40, mr = 8, mt = 22, mb = 40;
    const rows = st.rows.map((r) => ({ ...r, pc: (100 * r.p) / st.norm }));
    const ymax = Math.max(250, d3.max(rows, (r) => r.pc) * 1.05);
    const x = d3.scaleBand().domain(CAT_IDS).range([ml, w - mr]).padding(0.12);
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([h - mb, mt]);
    yGrid(svg, y, ml, w - mr, y.ticks(5));
    svg.append('line').attr('x1', ml).attr('x2', w - mr).attr('y1', y(100)).attr('y2', y(100)).style('stroke', 'var(--ink-2)').style('stroke-width', 1);
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + '%').tickSize(0).tickPadding(6))).select('.domain').remove();
    const hash = (Y) => { let t = Math.sin(Y * 12.9898) * 43758.5453; return t - Math.floor(t); };
    const g = svg.append('g');
    for (const c of CAT_IDS) {
      const rs = rows.filter((r) => r.cat === c);
      const cx = x(c) + x.bandwidth() / 2, jw = x.bandwidth() * 0.32;
      g.selectAll(null).data(rs).join('circle')
        .attr('cx', (r) => cx + (hash(r.Y) - 0.5) * 2 * jw).attr('cy', (r) => y(r.pc)).attr('r', 3.8)
        .style('fill', (r) => (r.off ? catVar(c) : 'var(--surface)')).style('stroke', (r) => (r.off ? 'var(--surface)' : catVar(c))).style('stroke-width', (r) => (r.off ? 1.2 : 1.8))
        .on('pointermove', (e, r) => showTip(e, sLabel(r.Y), [{ value: f0(r.pc) + '%', label: ` of normal (${f1(r.p)} in)` }, { value: signed(r.x) + ' °C', label: ' ' + CAT_LABEL[c], color: catVar(c) }]))
        .on('pointerleave', hideTip);
      if (rs.length) {
        const md = d3.median(rs, (r) => r.pc);
        g.append('line').attr('x1', cx - x.bandwidth() * 0.38).attr('x2', cx + x.bandwidth() * 0.38).attr('y1', y(md)).attr('y2', y(md)).style('stroke', 'var(--ink)').style('stroke-width', 2.5).style('stroke-linecap', 'round');
        const k = rs.filter((r) => r.pc > 100).length;
        text(g, cx, mt - 8, `${k} of ${rs.length}`, { anchor: 'middle', size: 10.5, fill: 'var(--ink-2)', weight: 600 });
      }
      const [a, b] = CAT_SHORT[c];
      text(g, cx, h - mb + 15, a, { anchor: 'middle', size: 10.5 });
      if (b) text(g, cx, h - mb + 28, b, { anchor: 'middle', size: 10.5 });
    }
    text(svg, ml - 4, mt - 8, 'above normal:', { anchor: 'end', size: 10, fill: 'var(--muted)' });

    // takeaway callout
    const strong = rows.filter((r) => r.x >= 1.5), oth = rows.filter((r) => r.x < 0.5), ln = rows.filter((r) => r.x <= -0.5);
    const box = $('takeaway'); box.replaceChildren();
    const add = (tag, str) => { const e = document.createElement(tag); e.textContent = str; box.append(e); return e; };
    add('strong', `${PLACE[state.place].name}, ${SEASON_SHORT[state.season]}, ${periodLabel(st.m)}`);
    if (strong.length) {
      const k = strong.filter((r) => r.pc > 100).length;
      const lo = strong.reduce((a, b) => (b.pc < a.pc ? b : a)), hi = strong.reduce((a, b) => (b.pc > a.pc ? b : a));
      add('p', `Strong and very strong El Niño seasons: ${k} of ${strong.length} were wetter than normal. Median ${f0(d3.median(strong, (r) => r.pc))}% of normal, ranging from ${f0(lo.pc)}% (${sLabel(lo.Y)}) to ${f0(hi.pc)}% (${sLabel(hi.Y)}).`);
    }
    if (oth.length) add('p', `Seasons without El Niño: ${oth.filter((r) => r.pc > 100).length} of ${oth.length} wetter than normal, median ${f0(d3.median(oth, (r) => r.pc))}%.`);
    if (ln.length) add('p', `La Niña seasons alone: ${ln.filter((r) => r.pc > 100).length} of ${ln.length} wetter than normal, median ${f0(d3.median(ln, (r) => r.pc))}%.`);
    add('p', 'Because a few very wet years pull the average up, most seasons fall below the "normal" average. That is why fewer than half of all seasons count as above normal.').className = 'small';
  }

  // ------------------------------------------------------------------ history: time series
  function renderTS(st) {
    const P = PLACE[state.place];
    const rows = st.rows;
    const el = $('ts');
    const H = el.clientWidth < 560 ? 230 : 280;
    const { svg, w, h } = prep('ts', H, `Season precipitation by year at ${P.name}, with the 2026–27 projection`);
    const ml = 44, mr = 12, mt = 16, mb = 26;
    const yrs = d3.range(st.m.y0, st.m.y1 + 1);
    const dom = [...yrs, 'gap', 'gap2', 2027];
    const x = d3.scaleBand().domain(dom).range([ml, w - mr]).paddingInner(0.15);
    const ymax = Math.max(d3.max(rows, (r) => r.p), st.q90) * 1.06;
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([h - mb, mt]);
    yGrid(svg, y, ml, w - mr, y.ticks(5));
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(unitTop(y, 5)).tickSize(0).tickPadding(6))).select('.domain').remove();
    const bw = Math.min(24, x.bandwidth()), off = (x.bandwidth() - bw) / 2;
    svg.append('g').selectAll('path').data(rows).join('path')
      .attr('d', (r) => barPath(x(r.Y) + off, y(0), bw, y(r.p), Math.min(3, bw / 2)))
      .style('fill', (r) => catVar(r.cat))
      .on('pointermove', (e, r) => showTip(e, sLabel(r.Y), [{ value: `${f1(r.p)} in`, label: ` ${f0((100 * r.p) / st.norm)}% of normal` }, { value: signed(r.x) + ' °C', label: ' ' + CAT_LABEL[r.cat], color: catVar(r.cat) }]))
      .on('pointerleave', hideTip);
    svg.append('line').attr('x1', ml).attr('x2', x(st.m.y1) + x.bandwidth()).attr('y1', y(st.norm)).attr('y2', y(st.norm)).style('stroke', 'var(--ink-2)').style('stroke-width', 1);
    // projection box
    const cx = x(2027) + x.bandwidth() / 2, bwid = Math.max(10, Math.min(22, x.bandwidth() * 3));
    const g = svg.append('g');
    g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(st.q90)).attr('y2', y(st.q10)).style('stroke', 'var(--fc)').style('stroke-width', 2);
    g.append('rect').attr('x', cx - bwid / 2).attr('width', bwid).attr('y', y(st.q75)).attr('height', y(st.q25) - y(st.q75)).attr('rx', 2)
      .style('fill', 'var(--fc-wash-2)').style('stroke', 'var(--fc)').style('stroke-width', 1.5);
    g.append('line').attr('x1', cx - bwid / 2 - 3).attr('x2', cx + bwid / 2 + 3).attr('y1', y(st.med)).attr('y2', y(st.med)).style('stroke', 'var(--fc)').style('stroke-width', 3);
    g.append('rect').attr('x', cx - 18).attr('width', 36).attr('y', y(st.q90) - 6).attr('height', y(st.q10) - y(st.q90) + 12).style('fill', 'transparent')
      .on('pointermove', (e) => showTip(e, '2026–27 projection', [{ value: `${inch(st.med)} in`, label: ' middle estimate' }, { value: `${inch(st.q25)}–${inch(st.q75)} in`, label: ' middle 50%' }, { value: `${inch(st.q10)}–${inch(st.q90)} in`, label: ' 8 in 10' }]))
      .on('pointerleave', hideTip);
    text(svg, cx, h - 8, '2026–27', { anchor: 'end', size: 11, fill: 'var(--fc-text)', weight: 700 });
    const step = yrs.length > 100 ? 20 : 10;
    yrs.filter((Y) => (Y - 1) % step === 0 && x(Y) != null && x(2027) - x(Y) > 60).forEach((Y) => text(svg, x(Y) + x.bandwidth() / 2, h - 8, String(Y - 1), { anchor: 'middle', size: 11 }));
    $('ts-sub').textContent = `${P.name}, ${SEASON_SHORT[state.season]}, ${periodLabel(st.m)}. Bars are colored by that winter's El Niño–La Niña state; the horizontal line is the 1991–2020 normal; the teal box at right is the 2026–27 projection (box: middle 50%, whiskers: 8 in 10).`;
    legendCats('cat-legend-1'); legendCats('cat-legend-2');

    // season table
    const tbl = $('season-table'); tbl.replaceChildren();
    const hr = tbl.createTHead().insertRow();
    ['Season', 'Nov–Jan RONI', 'Category', 'Total (in)', '% of normal'].forEach((t, i) => { const th = document.createElement('th'); th.textContent = t; if (i === 1 || i > 2) th.className = 'n'; hr.append(th); });
    const tb = tbl.createTBody();
    rows.slice().reverse().forEach((r) => {
      const tr = tb.insertRow();
      [sLabel(r.Y), signed(r.x) + (r.off ? '' : '*'), CAT_LABEL[r.cat], f2(r.p), f0((100 * r.p) / st.norm)].forEach((t, i) => {
        const c = tr.insertCell(); c.textContent = t; if (i === 1 || i > 2) c.className = 'n';
      });
    });
    const cap = tbl.createCaption(); cap.className = 'small'; cap.style.textAlign = 'left'; cap.style.captionSide = 'bottom';
    cap.textContent = '* El Niño index reconstructed from ERSSTv5 (before 1950).';
  }

  // ------------------------------------------------------------------ this event: trajectories
  function renderTraj() {
    const T = D.traj;
    const el = $('traj');
    const H = el.clientWidth < 560 ? 300 : 360;
    const { svg, w, h } = prep('traj', H, 'Relative Oceanic Niño Index through past strong El Niño events and 2026 with the NOAA forecast');
    const ml = 34, mr = 16, mt = 16, mb = 42;
    const n = T.months.length;
    const x = d3.scalePoint().domain(d3.range(n)).range([ml, w - mr]);
    const y = d3.scaleLinear().domain([-1.2, 3.6]).range([h - mb, mt]);
    yGrid(svg, y, ml, w - mr, [-1, 0, 1, 2, 3]);
    [[1.5, 'Strong'], [2.0, 'Very strong'], [2.5, 'Beyond any since 1950']].forEach(([v, lab]) => {
      svg.append('line').attr('x1', ml).attr('x2', w - mr).attr('y1', y(v)).attr('y2', y(v)).style('stroke', 'var(--axis)').style('stroke-width', 1);
      text(svg, ml + 4, y(v) - 4, lab, { size: 10.5, fill: 'var(--muted)', halo: true });
    });
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).tickValues([-1, 0, 1, 2, 3]).tickFormat((d) => (d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d)).tickSize(0).tickPadding(6))).select('.domain').remove();
    const narrow = w < 560;
    T.months.forEach((mo, i) => { if (!narrow || i % 2 === 0) text(svg, x(i), h - mb + 15, mo, { anchor: 'middle', size: 10.5 }); });
    text(svg, x(0), h - 6, 'Year 0 (e.g. 1997, 2026)', { size: 10.5, fill: 'var(--ink-2)' });
    text(svg, x(12), h - 6, 'Year 1', { size: 10.5, fill: 'var(--ink-2)' });
    const line = d3.line().defined((d) => d != null).x((d, i) => x(i)).y((d) => y(d)).curve(d3.curveMonotoneX);
    const hi = new Set([1982, 1997, 2015]);
    const g = svg.append('g');
    T.events.forEach((ev) => {
      g.append('path').datum(ev.v).attr('d', line).style('fill', 'none').style('stroke', 'var(--ink-2)')
        .style('stroke-width', hi.has(ev.y0) ? 1.8 : 1.2).style('opacity', hi.has(ev.y0) ? 0.75 : 0.28);
    });
    // label the three very strong events at their peaks
    const placed = [];
    T.events.filter((e) => hi.has(e.y0)).forEach((ev) => {
      let bi = 0; ev.v.forEach((v, i) => { if (v != null && v > ev.v[bi]) bi = i; });
      let ly = y(ev.v[bi]) - 8;
      for (const p of placed) if (Math.abs(p - ly) < 12) ly = p - 13;
      placed.push(ly);
      text(g, x(bi) + (bi > n / 2 ? 6 : -6), ly, ev.label, { anchor: bi > n / 2 ? 'start' : 'end', size: 11, fill: 'var(--ink-2)', weight: 650, halo: true });
    });
    // forecast fan
    const fc = T.fc;
    const last = T.current.v.reduce((a, v, i) => (v != null ? i : a), 0);
    const anchor = { i: last, q05: T.current.v[last], q25: T.current.v[last], q50: T.current.v[last], q75: T.current.v[last], q95: T.current.v[last] };
    const fan = [anchor, ...fc];
    const area = (a, b) => d3.area().x((d) => x(d.i)).y0((d) => y(d[a])).y1((d) => y(d[b])).curve(d3.curveMonotoneX);
    svg.append('path').datum(fan).attr('d', area('q05', 'q95')).style('fill', 'var(--fc-wash)');
    svg.append('path').datum(fan).attr('d', area('q25', 'q75')).style('fill', 'var(--fc-wash-2)');
    svg.append('path').datum(fan).attr('d', d3.line().x((d) => x(d.i)).y((d) => y(d.q50)).curve(d3.curveMonotoneX))
      .style('fill', 'none').style('stroke', 'var(--fc)').style('stroke-width', 2).style('stroke-dasharray', '6 4');
    svg.append('path').datum(T.current.v).attr('d', line).style('fill', 'none').style('stroke', 'var(--fc)').style('stroke-width', 3.2);
    svg.append('g').selectAll('circle').data(T.current.v.map((v, i) => ({ v, i })).filter((d) => d.v != null)).join('circle')
      .attr('cx', (d) => x(d.i)).attr('cy', (d) => y(d.v)).attr('r', 3.5).style('fill', 'var(--fc)').style('stroke', 'var(--surface)').style('stroke-width', 1.5);
    const pk = fc.reduce((a, b) => (b.q50 > a.q50 ? b : a));
    text(svg, x(pk.i), y(pk.q95) - 8, `2026–27 forecast peak ${signed(pk.q50)}`, { anchor: 'middle', size: 12, fill: 'var(--fc-text)', weight: 750, halo: true });
    text(svg, x(last) + 8, y(T.current.v[last]) + 18, `${T.months[last]} 2026: ${signed(T.current.v[last])}`, { anchor: 'start', size: 11, fill: 'var(--fc-text)', weight: 700, halo: true });
    // hover crosshair
    const cross = svg.append('line').attr('y1', mt).attr('y2', h - mb).style('stroke', 'var(--axis)').style('opacity', 0);
    svg.append('rect').attr('x', ml).attr('y', mt).attr('width', w - ml - mr).attr('height', h - mb - mt).style('fill', 'transparent')
      .on('pointermove', (e) => {
        const [px] = d3.pointer(e);
        const i = Math.max(0, Math.min(n - 1, Math.round((px - ml) / x.step())));
        cross.attr('x1', x(i)).attr('x2', x(i)).style('opacity', 1);
        const rowsT = [];
        const f = fc.find((d) => d.i === i);
        if (T.current.v[i] != null) rowsT.push({ value: signed(T.current.v[i]), label: ' 2026 observed', color: 'var(--fc)' });
        else if (f) rowsT.push({ value: signed(f.q50), label: ` 2026–27 forecast (90%: ${signed(f.q05)} to ${signed(f.q95)})`, color: 'var(--fc)' });
        T.events.map((ev) => ({ ev, v: ev.v[i] })).filter((d) => d.v != null).sort((a, b) => b.v - a.v)
          .forEach((d) => rowsT.push({ value: signed(d.v), label: ' ' + d.ev.label, color: 'var(--ink-2)' }));
        showTip(e, `${T.months[i]} of year ${i < 12 ? 0 : 1}`, rowsT);
      })
      .on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
  }

  // ------------------------------------------------------------------ SST map
  let coastPaths = null;
  function renderSSTButtons() {
    const seg = $('seg-sst'); seg.replaceChildren();
    G.sst.maps.forEach((mp, i) => {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = mp.label;
      b.setAttribute('aria-pressed', String(i === state.sst));
      b.addEventListener('click', () => { state.sst = i; [...seg.children].forEach((c, j) => c.setAttribute('aria-pressed', String(j === i))); renderSST(); });
      seg.append(b);
    });
  }
  function renderSST() {
    const S = G.sst, mp = S.maps[state.sst];
    const cv = $('sst-canvas');
    const W = Math.max(260, cv.parentElement.clientWidth);
    const lon0 = S.lon[0] - 1, lon1 = S.lon[S.lon.length - 1] + 1, lat1 = S.lat[0] + 1, lat0 = S.lat[S.lat.length - 1] - 1;
    const sc = W / (lon1 - lon0), H = Math.round((lat1 - lat0) * sc);
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css('--surface'); ctx.fillRect(0, 0, W, H);
    const color = diverging(sstStops(), [-4, 0, 4]);
    const shift = state.sstRel ? mp.trop : 0;
    const X = (lon) => (lon - lon0) * sc, Y = (lat) => (lat1 - lat) * sc;
    for (let i = 0; i < S.lat.length; i++) {
      for (let j = 0; j < S.lon.length; j++) {
        const v = mp.grid[i][j];
        if (v == null) continue;
        ctx.fillStyle = color(v / 10 - shift);
        ctx.fillRect(X(S.lon[j] - 1), Y(S.lat[i] + 1), 2 * sc + 0.6, 2 * sc + 0.6);
      }
    }
    if (!coastPaths) coastPaths = S.coast.map((d) => new Path2D(d));
    ctx.save(); ctx.translate(X(120), Y(60)); ctx.scale(sc, sc);
    ctx.fillStyle = css('--land'); ctx.strokeStyle = css('--axis'); ctx.lineWidth = 0.8 / sc;
    coastPaths.forEach((p) => { ctx.fill(p); ctx.stroke(p); });
    ctx.restore();
    // Niño-3.4 box and graticule labels
    ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.5;
    ctx.strokeRect(X(190), Y(5), X(240) - X(190), Y(-5) - Y(5));
    ctx.font = `600 ${W < 560 ? 10 : 12}px ${css('--font-display') || 'sans-serif'}`;
    ctx.fillStyle = css('--ink'); ctx.textAlign = 'left';
    ctx.fillText('Niño‑3.4', X(190) + 2, Y(5) - 4);
    ctx.fillStyle = css('--muted'); ctx.font = `500 ${W < 560 ? 9 : 11}px ${css('--font-display') || 'sans-serif'}`;
    ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1;
    [[0, 'Equator'], [30, '30°N'], [-20, '20°S']].forEach(([la, lab]) => { ctx.fillText(lab, 4, Y(la) - 3); });
    [[180, '180°'], [210, '150°W'], [240, '120°W']].forEach(([lo, lab]) => { ctx.textAlign = 'center'; ctx.fillText(lab, X(lo), H - 5); });
    ctx.textAlign = 'right'; ctx.fillStyle = css('--ink-2');
    ctx.fillText('California', X(242) - 4, Y(36));
    $('sst-cap').textContent = `${mp.label}: Niño‑3.4 ${signed(mp.n34)} °C, tropical average ${signed(mp.trop)} °C${mp.ndj != null ? `, winter peak (Nov–Jan RONI) ${signed(mp.ndj)} °C` : ''}`;
    cv.setAttribute('aria-label', `Sea-surface temperature anomaly map, ${mp.label}. Niño‑3.4 anomaly ${signed(mp.n34)} °C.`);
    cv.onpointermove = (e) => {
      const r = cv.getBoundingClientRect(); const lon = lon0 + (e.clientX - r.left) / sc, lat = lat1 - (e.clientY - r.top) / sc;
      const i = Math.round((S.lat[0] - lat) / 2), j = Math.round((lon - S.lon[0]) / 2);
      if (i < 0 || j < 0 || i >= S.lat.length || j >= S.lon.length || mp.grid[i][j] == null) { hideTip(); return; }
      const lonLab = S.lon[j] <= 180 ? `${S.lon[j]}°E` : `${360 - S.lon[j]}°W`, latLab = S.lat[i] >= 0 ? `${S.lat[i]}°N` : `${-S.lat[i]}°S`;
      showTip(e, `${latLab}, ${lonLab}`, [{ value: signed(mp.grid[i][j] / 10 - shift, f1) + ' °C', label: state.sstRel ? ' minus tropical average' : ' vs 1991–2020' }]);
    };
    cv.onpointerleave = hideTip;
    drawLegend('sst-legend', color, [-4, 4]);
  }
  function drawLegend(id, scale, [a, b]) {
    const c = $(id), ctx = c.getContext('2d');
    for (let i = 0; i < c.width; i++) { ctx.fillStyle = scale(a + ((b - a) * i) / (c.width - 1)); ctx.fillRect(i, 0, 1, c.height); }
  }

  // ------------------------------------------------------------------ timing: monthly composites
  function renderMonthly() {
    const P = PLACE[state.place];
    const M = D.monthly[state.place][state.period];
    const months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'];
    const comb = (ids) => {
      const tot = ids.reduce((a, c) => a + M[c].n, 0);
      if (!tot) return null;
      return { n: tot, mean: months.map((_, i) => ids.reduce((a, c) => a + (M[c].mean[i] || 0) * M[c].n, 0) / tot) };
    };
    const groups = [
      { id: 'enVS', lab: 'Very strong El Niño', col: catVar('enVS'), d: comb(['enVS']) },
      { id: 'enS', lab: 'Strong El Niño', col: catVar('enS'), d: comb(['enS']) },
      { id: 'neu', lab: 'Neutral', col: catVar('neu'), d: comb(['neu']) },
      { id: 'ln', lab: 'La Niña', col: catVar('lnS'), d: comb(['lnS', 'lnW']) },
    ].filter((g) => g.d);
    const leg = $('monthly-legend'); leg.replaceChildren();
    groups.forEach((g) => { const s = document.createElement('span'); const i = document.createElement('i'); i.className = 'line'; i.style.background = g.col; s.append(i, document.createTextNode(`${g.lab} (${g.d.n})`)); leg.append(s); });
    const el = $('monthly');
    const { svg, w, h } = prep('monthly', el.clientWidth < 420 ? 240 : 270, `Average monthly precipitation by El Niño–La Niña group at ${P.name}`);
    const ml = 42, mr = 10, mt = 14, mb = 24;
    const x = d3.scalePoint().domain(d3.range(12)).range([ml, w - mr]).padding(0.3);
    const ymax = d3.max(groups, (g) => d3.max(g.d.mean));
    const y = d3.scaleLinear().domain([0, ymax * 1.1]).nice().range([h - mb, mt]);
    yGrid(svg, y, ml, w - mr, y.ticks(5));
    styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(unitTop(y, 5)).tickSize(0).tickPadding(6))).select('.domain').remove();
    months.forEach((mo, i) => text(svg, x(i), h - 6, w < 420 ? mo[0] : mo, { anchor: 'middle', size: 10.5 }));
    const line = d3.line().x((d, i) => x(i)).y((d) => y(d)).curve(d3.curveMonotoneX);
    groups.slice().reverse().forEach((g) => {
      svg.append('path').datum(g.d.mean).attr('d', line).style('fill', 'none').style('stroke', g.col).style('stroke-width', 2);
      svg.append('g').selectAll('circle').data(g.d.mean).join('circle').attr('cx', (d, i) => x(i)).attr('cy', (d) => y(d)).attr('r', 3.2)
        .style('fill', g.col).style('stroke', 'var(--surface)').style('stroke-width', 1.5);
    });
    const cross = svg.append('line').attr('y1', mt).attr('y2', h - mb).style('stroke', 'var(--axis)').style('opacity', 0);
    svg.append('rect').attr('x', ml).attr('y', mt).attr('width', w - ml - mr).attr('height', h - mb - mt).style('fill', 'transparent')
      .on('pointermove', (e) => {
        const [px] = d3.pointer(e); const i = Math.max(0, Math.min(11, Math.round((px - x(0)) / x.step())));
        cross.attr('x1', x(i)).attr('x2', x(i)).style('opacity', 1);
        showTip(e, months[i], groups.map((g) => ({ value: f2(g.d.mean[i]) + ' in', label: ' ' + g.lab, color: g.col })));
      })
      .on('pointerleave', () => { cross.style('opacity', 0); hideTip(); });
    $('monthly-sub').textContent = `${P.name}, ${periodLabel(model())}. Average precipitation in each month of the water year, grouped by that winter's El Niño–La Niña state (number of seasons in parentheses).`;
    const tbl = $('monthly-table'); tbl.replaceChildren();
    const hr = tbl.createTHead().insertRow();
    ['Group', ...months].forEach((t, i) => { const th = document.createElement('th'); th.textContent = t; if (i) th.className = 'n'; hr.append(th); });
    const tb = tbl.createTBody();
    groups.forEach((g) => { const tr = tb.insertRow(); [g.lab, ...g.d.mean.map((v) => f2(v))].forEach((t, i) => { const c = tr.insertCell(); c.textContent = t; if (i) c.className = 'n'; }); });
  }

  // ------------------------------------------------------------------ timing: heavy-rain days
  function renderExtremes() {
    const pid = PLACE[state.place].kind === 'station' ? state.place : state.place === 'desert' ? 'ps' : 'la';
    const P = PLACE[pid];
    const E = D.extremes[pid][state.season];
    const m = model(pid, state.season, state.period);
    const groups = [
      { id: 'ln', lab: ['La Niña', ''], test: (x) => x <= -0.5, col: catVar('lnS') },
      { id: 'neu', lab: ['Neutral', ''], test: (x) => x > -0.5 && x < 0.5, col: catVar('neu') },
      { id: 'enWM', lab: ['Weak–mod.', 'El Niño'], test: (x) => x >= 0.5 && x < 1.5, col: catVar('enWM') },
      { id: 'enS', lab: ['Strong+', 'El Niño'], test: (x) => x >= 1.5, col: catVar('enVS') },
    ];
    const pts = [];
    YEARS.forEach((Y, i) => {
      if (Y < m.y0 || Y > m.y1 || E.d1[i] == null || NDJ[i] == null) return;
      const g = groups.find((gg) => gg.test(NDJ[i]));
      pts.push({ Y, x: NDJ[i], d1: E.d1[i], mx: E.max1[i], g });
    });
    const hash = (Y) => { const t = Math.sin(Y * 78.233) * 43758.5453; return t - Math.floor(t); };
    function strip(id, key, title, fmt) {
      const { svg, w, h } = prep(id, 210, title);
      const ml = 34, mr = 26, mt = 26, mb = 36;
      const x = d3.scaleBand().domain(groups.map((g) => g.id)).range([ml, w - mr]).padding(0.15);
      const y = d3.scaleLinear().domain([0, d3.max(pts, (p) => p[key]) * 1.08 || 1]).nice().range([h - mb, mt]);
      yGrid(svg, y, ml, w - mr, y.ticks(4));
      styleAxis(svg.append('g').attr('transform', `translate(${ml},0)`).call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6))).select('.domain').remove();
      text(svg, 0, 12, title, { size: 12, fill: 'var(--ink)', weight: 700 });
      groups.forEach((g) => {
        const ps = pts.filter((p) => p.g === g), cx = x(g.id) + x.bandwidth() / 2, jw = x.bandwidth() * 0.3;
        const dots = svg.append('g');
        if (ps.length) {
          const mean = d3.mean(ps, (p) => p[key]);
          svg.append('line').attr('x1', cx - x.bandwidth() * 0.36).attr('x2', cx + x.bandwidth() * 0.36).attr('y1', y(mean)).attr('y2', y(mean)).style('stroke', 'var(--ink)').style('stroke-width', 2.5).style('stroke-linecap', 'round');
          text(svg, cx + x.bandwidth() * 0.38 + 3, y(mean) + 4, fmt(mean), { size: 10.5, fill: 'var(--ink)', weight: 700, anchor: 'start', halo: true });
        }
        dots.selectAll('circle').data(ps).join('circle')
          .attr('cx', (p) => cx + (hash(p.Y) - 0.5) * 2 * jw).attr('cy', (p) => y(p[key])).attr('r', 3.4)
          .style('fill', g.col).style('stroke', 'var(--surface)').style('stroke-width', 1).style('opacity', 0.85)
          .on('pointermove', (e, p) => showTip(e, sLabel(p.Y), [{ value: key === 'd1' ? `${p.d1} days` : `${f2(p.mx)} in`, label: key === 'd1' ? ' with 1 inch or more' : ' on the wettest day' }, { value: signed(p.x) + ' °C', label: ' ' + CAT_LABEL[catOf(p.x)], color: g.col }]))
          .on('pointerleave', hideTip);
        text(svg, cx, h - mb + 14, g.lab[0], { anchor: 'middle', size: 10.5 });
        if (g.lab[1]) text(svg, cx, h - mb + 27, g.lab[1], { anchor: 'middle', size: 10.5 });
      });
    }
    strip('ext1', 'd1', 'Days with 1 inch or more', (v) => f1(v));
    strip('ext2', 'mx', 'Wettest single day (inches)', (v) => f2(v));
    $('ext-sub').textContent = `${P.name}${P.id !== state.place ? ' (station for this region)' : ''}, ${SEASON_SHORT[state.season]}, ${periodLabel(m)}. Dots are seasons; bars and numbers show group averages.`;
  }

  // ------------------------------------------------------------------ nationwide map
  function renderMap() {
    const M = G.divmap;
    const el = $('usmap');
    el.replaceChildren();
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${M.w} ${M.h}`).attr('width', '100%').attr('role', 'img')
      .attr('aria-label', 'Map of U.S. climate divisions showing precipitation in strong El Niño seasons');
    const idx = state.metric;
    const scales = [
      { s: d3.scaleDivergingLog(d3.interpolateRgbBasis(wetStops())).domain([40, 100, 250]).clamp(true), lo: '40%', hi: '250%', fmt: (v) => f0(v) + '% of normal', ttl: '% of normal precipitation in strong El Niño seasons' },
      { s: diverging(wetStops(), [0, 0.5, 1]), lo: '0 of 8', hi: '8 of 8', fmt: (v) => `${Math.round(v * 8)} of 8 seasons wetter than normal`, ttl: 'How often strong El Niño seasons were wetter than normal' },
      { s: diverging(wetStops(), [-0.6, 0, 0.6]), lo: '−0.6', hi: '+0.6', fmt: (v) => 'rank correlation ' + signed(v), ttl: 'Correlation between El Niño strength and precipitation, all seasons' },
    ][idx];
    const g = svg.append('g');
    g.selectAll('path').data(M.divs).join('path').attr('d', (d) => d.d)
      .style('fill', (d) => { const v = d.v[state.season][idx]; return v == null ? 'var(--land)' : scales.s(v); })
      .style('stroke', 'var(--surface)').style('stroke-width', 0.4)
      .on('pointermove', function (e, d) {
        d3.select(this).style('stroke', 'var(--ink)').style('stroke-width', 1.2).raise();
        const v = d.v[state.season];
        showTip(e, `${d.name}, ${d.st}`, [{ value: scales.fmt(v[idx]), label: '' }, ...(idx !== 0 ? [{ value: f0(v[0]) + '%', label: ' of normal, strong El Niño average' }] : [])]);
      })
      .on('pointerleave', function () { d3.select(this).style('stroke', 'var(--surface)').style('stroke-width', 0.4); hideTip(); });
    svg.append('path').attr('d', M.states).style('fill', 'none').style('stroke', 'var(--surface)').style('stroke-width', 1.6).style('pointer-events', 'none');
    svg.append('g').selectAll('path').data(M.divs.filter((d) => d.code === 406 || d.code === 407)).join('path').attr('d', (d) => d.d)
      .style('fill', 'none').style('stroke', 'var(--ink)').style('stroke-width', 1.6).style('pointer-events', 'none');
    $('map-title').textContent = scales.ttl;
    $('map-sub').textContent = `${D.seasons[state.season].label} (${SEASON_SHORT[state.season]}), 1951–2026. Brown means drier than normal, green means wetter.`;
    $('map-lo').textContent = scales.lo; $('map-hi').textContent = scales.hi;
    const c = $('map-legend'), ctx = c.getContext('2d');
    const dom = idx === 0 ? [40, 250] : idx === 1 ? [0, 1] : [-0.6, 0.6];
    for (let i = 0; i < c.width; i++) {
      const t = i / (c.width - 1);
      const v = idx === 0 ? Math.exp(Math.log(dom[0]) + t * (Math.log(dom[1]) - Math.log(dom[0]))) : dom[0] + t * (dom[1] - dom[0]);
      ctx.fillStyle = scales.s(v); ctx.fillRect(i, 0, 1, c.height);
    }
  }

  // ------------------------------------------------------------------ case files
  function renderCases(st) {
    const P = PLACE[state.place];
    const m = st.m;
    const hind = m.hind.filter((r) => r[1] >= 1.5).sort((a, b) => b[1] - a[1]);
    const tbl = $('cases-table'); tbl.replaceChildren();
    const hr = tbl.createTHead().insertRow();
    ['Season', 'Strength (RONI)', 'Traditional ONI', 'Observed', 'Model said (middle, 8 in 10)', 'Outcome'].forEach((t, i) => { const th = document.createElement('th'); th.textContent = t; if (i > 0 && i < 5) th.className = 'n'; hr.append(th); });
    const tb = tbl.createTBody();
    hind.forEach(([Y, x, obs, q10, q50, q90]) => {
      const tr = tb.insertRow();
      const oni = D.oni[YI(Y)];
      const cells = [sLabel(Y), signed(x) + (Y < 1950 ? '*' : ''), oni == null ? '–' : signed(oni), `${f1(obs)} in · ${f0((100 * obs) / st.norm)}%`, `${inch(q50)} in (${inch(q10)}–${inch(q90)})`];
      cells.forEach((t, i) => { const c = tr.insertCell(); c.textContent = t; if (i > 0) c.className = 'n'; });
      const c = tr.insertCell(); const pill = document.createElement('span');
      if (obs > q90) { pill.className = 'pill wet'; pill.textContent = 'Wetter than range'; }
      else if (obs < q10) { pill.className = 'pill dry'; pill.textContent = 'Drier than range'; }
      else { pill.className = 'pill in'; pill.textContent = 'Inside range'; }
      c.append(pill);
    });
    const all = m.hind;
    const k = all.filter((r) => r[6] >= 0.1 && r[6] <= 0.9).length;
    $('cases-title').textContent = `Strong El Niño seasons at ${P.name}, ${SEASON_SHORT[state.season]}`;
    $('cases-sub').textContent = `Seasons with a November–January relative index of +1.5 °C or more, ${periodLabel(m)}, strongest first. “Model said” is the leave-one-out hindcast from El Niño strength alone.${hind.some((r) => r[0] < 1950) ? ' * Index reconstructed from ERSSTv5.' : ''}`;
    const share = k / all.length;
    $('cases-foot').textContent = `Across all ${all.length} El Niño seasons with an index of +1.0 °C or more in this record, ${k} landed inside the model's 8‑in‑10 range (${pct(share)}). ` +
      (share < 0.75 ? 'A perfectly calibrated method would capture about 8 in 10, so treat the ranges on this page as a lower bound on the real uncertainty.' : 'That is close to the 8 in 10 a well-calibrated method should capture.');
  }

  // ------------------------------------------------------------------ methods text
  function renderMethods(st) {
    const sk = st.m.skill;
    const P = PLACE[state.place];
    const p = (v) => (v == null ? '–' : `${v >= 0 ? '' : '−'}${f0(Math.abs(100 * v))}%`);
    $('val-text').textContent = `Leave-one-out cross-validation compares the model with a climatology that ignores El Niño, using the continuous ranked probability score (CRPS) [25, 26]. For ${P.name}, ${SEASON_SHORT[state.season]}, ${periodLabel(st.m)}, the skill over climatology is ${p(sk.all)} across all seasons, ${p(sk.en)} in El Niño seasons and ${p(sk.strong)} in strong El Niño seasons. Positive numbers mean the model beats climatology. Modest values like these are normal for seasonal rainfall, which is dominated by unpredictable weather.`;
    document.querySelectorAll('[data-bind="pdo"]').forEach((e) => { e.textContent = signed(D.current.pdo_latest); });
    document.querySelectorAll('[data-bind="recon-r"]').forEach((e) => { e.textContent = d3.format('.3f')(D.meta.recon.r); });
    document.querySelectorAll('[data-bind="recon-rmse"]').forEach((e) => { e.textContent = f2(D.meta.recon.rmse); });
    document.querySelectorAll('[data-bind="nboot"]').forEach((e) => { e.textContent = String(D.meta.nboot); });
    const fill = D.meta.fills && D.meta.fills.sb;
    document.querySelectorAll('[data-bind="sb-fill"]').forEach((e) => { e.textContent = fill ? String(fill.months) : '–'; });
  }

  // ------------------------------------------------------------------ controls and update loop
  function setMu(v) {
    state.mu = Math.round(v * 100) / 100;
    $('mu').value = String(Math.min(3.8, Math.max(0.5, state.mu)));
    syncOutputs();
    schedule('fc');
  }
  function syncOutputs() {
    $('mu-out').textContent = signed(state.mu) + ' °C';
    $('sigma-out').textContent = '± ' + f2(state.sigma) + ' °C';
  }
  function initControls() {
    const sel = $('place');
    const groups = [['Regions (NOAA climate divisions)', 'region'], ['Cities and towns (weather stations)', 'station']];
    for (const [lab, kind] of groups) {
      const og = document.createElement('optgroup'); og.label = lab;
      D.places.filter((p) => p.kind === kind).forEach((p) => { const o = document.createElement('option'); o.value = p.id; o.textContent = PLACE_LABEL[p.id] || p.name; o.title = p.sub; og.append(o); });
      sel.append(og);
    }
    sel.value = state.place;
    sel.addEventListener('change', () => { state.place = sel.value; schedule('all'); });
    const ss = $('season');
    Object.entries(D.seasons).forEach(([k, s]) => { const o = document.createElement('option'); o.value = k; o.textContent = `${s.label} (${s.months})`; ss.append(o); });
    ss.value = state.season;
    ss.addEventListener('change', () => { state.season = ss.value; schedule('all'); });
    $('mu').addEventListener('input', (e) => { state.mu = +e.target.value; syncOutputs(); schedule('fc'); });
    $('sigma').addEventListener('input', (e) => { state.sigma = +e.target.value; syncOutputs(); schedule('fc'); });
    $('reset').addEventListener('click', () => { state.sigma = FC.ndj_sigma; $('sigma').value = String(state.sigma); setMu(FC.ndj_median); });
    const seg = (id, key, conv = (v) => v, scope = 'all', after) => {
      const el = $(id);
      el.addEventListener('click', (e) => {
        const b = e.target.closest('button'); if (!b) return;
        state[key] = conv(b.dataset.v);
        [...el.querySelectorAll('button')].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
        if (after) after(); else schedule(scope);
      });
    };
    seg('seg-period', 'period');
    seg('seg-extrap', 'extrap');
    seg('seg-sstrel', 'sstRel', (v) => v === '1', 'none', renderSST);
    seg('seg-metric', 'metric', (v) => +v, 'none', renderMap);
    $('mu').value = String(state.mu); $('sigma').value = String(state.sigma);
    syncOutputs();
    renderPresets();
    renderSSTButtons();
  }

  const pending = new Set();
  let raf = 0;
  function schedule(scope) { pending.add(scope); if (!raf) raf = requestAnimationFrame(flush); }
  function flush() {
    raf = 0;
    const all = pending.has('all');
    pending.clear();
    const st = outlookStats();
    renderTiles(st);
    renderStripes();
    renderRuler();
    renderOutlook(st);
    renderScatter(st);
    renderTS(st);
    if (all) {
      renderCats(st);
      renderTraj();
      renderSST();
      renderMonthly();
      renderExtremes();
      renderMap();
      renderCases(st);
      renderMethods(st);
    }
  }

  initControls();
  schedule('all');
  let lastW = window.innerWidth;
  let rt = 0;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (Math.abs(window.innerWidth - lastW) > 4) { lastW = window.innerWidth; schedule('all'); } }, 160);
  });
  const rerender = () => schedule('all');
  try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerender); } catch (e) { /* older browsers */ }
  new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
