/* ==================================================================== *
 *  Interactive figures of the deck                                     *
 *                                                                      *
 *  <div data-widget="plot">       convergence plots from the paper CSVs *
 *  <div data-widget="matrix">     the robustness summary               *
 *  <div data-widget="lab">        the stress-free state lab            *
 *  <div data-widget="transient">  the time scrubber                    *
 *  <div data-widget="rods">       rods under a non-symmetric stress    *
 *                                                                      *
 *  Data: window.PAPER, window.LAB (data.js), window.TRANSIENT,         *
 *  window.TORQUE (torque.js).                                          *
 *  The engine calls WIDGETS.show(host, step) whenever the host's slide  *
 *  is rendered, WIDGETS.leave(slide) when it is left.                  *
 * ==================================================================== */
window.WIDGETS = (function(){
'use strict';

/* ---------- small helpers ------------------------------------------ */
const NS = 'http://www.w3.org/2000/svg';
function h(tag, cls, html){
  const n = document.createElement(tag);
  if(cls) n.className = cls;
  if(html != null) n.innerHTML = html;
  return n;
}
function S(tag, attrs, parent){
  const n = document.createElementNS(NS, tag);
  for(const k in attrs) n.setAttribute(k, attrs[k]);
  if(parent) parent.appendChild(n);
  return n;
}
const SUP = {'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','.':'·'};
const sup = n => String(n).split('').map(c=>SUP[c] || c).join('');
function sci(x, d){
  d = d || 2;
  if(!isFinite(x)) return '–';
  if(x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)) + 1e-12);
  if(e >= -1 && e <= 2) return String(Number(x.toPrecision(d)));
  const m = x / Math.pow(10, e);
  return m.toFixed(d - 1) + '×10' + sup(e);
}
const dec = e => '10' + sup(e);
function b64bytes(s){
  const bin = atob(s), u = new Uint8Array(bin.length);
  for(let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
const f32 = s => s == null ? null : new Float32Array(b64bytes(s).buffer);
const u32 = s => new Uint32Array(b64bytes(s).buffer);
const clamp = (x,a,b) => Math.max(a, Math.min(b, x));

let PAL = null;
function pal(){
  if(PAL) return PAL;
  const cs = getComputedStyle(document.documentElement);
  const g = n => cs.getPropertyValue(n).trim();
  return PAL = {paper:g('--paper'), panel:g('--panel'), ink:g('--ink'), ink2:g('--ink2'),
    muted:g('--muted'), faint:g('--faint'), rule:g('--rule'), rule2:g('--rule2'),
    strong:g('--strong'), weak:g('--weak'), poll:g('--poll'), approx:g('--approx'),
    red:g('--red'), accent:g('--accent'), d:[g('--d1'), g('--d2'), g('--d3')]};
}
/* a colour t of the way from palette colour a to b, both #rrggbb */
function mixc(a, b, t){
  const p = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return 'rgb(' + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',') + ')';
}
/* the deck is scaled with a CSS transform: mouse positions come back in
   screen pixels, layout in slide pixels */
function local(evt, node){
  const r = node.getBoundingClientRect();
  const k = node.offsetWidth ? node.offsetWidth / r.width : 1;
  return [(evt.clientX - r.left) * k, (evt.clientY - r.top) * k];
}
function slideScale(){
  return parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--s')) || 1;
}
function sizeCanvas(cv, w, hgt){
  const dpr = Math.min(3, (window.devicePixelRatio || 1) * Math.max(1, slideScale()));
  cv.style.width = w + 'px'; cv.style.height = hgt + 'px';
  const W = Math.round(w * dpr), H = Math.round(hgt * dpr);
  if(cv.width !== W || cv.height !== H){ cv.width = W; cv.height = H; }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, hgt);
  return c;
}
function text(c, x, y, s, col, size, align, base, font){
  c.fillStyle = col;
  c.font = (font || '') + (size || 12) + 'px ui-sans-serif,-apple-system,Helvetica,Arial,sans-serif';
  c.textAlign = align || 'center'; c.textBaseline = base || 'middle';
  c.fillText(s, x, y);
}
function seg(labels, onpick, cls){
  const s = h('div', 'seg' + (cls ? ' ' + cls : ''));
  const bs = labels.map(([key, lab]) => {
    const b = h('button', null, lab);
    b.dataset.k = key;
    b.addEventListener('click', e => { e.stopPropagation(); onpick(key); });
    s.appendChild(b);
    return b;
  });
  s.set = key => bs.forEach(b => b.classList.toggle('on', b.dataset.k === String(key)));
  s.enable = (key, on) => bs.forEach(b => { if(b.dataset.k === String(key)) b.disabled = !on; });
  return s;
}
function katexInto(node, tex){
  if(window.katex) katex.render(tex, node, {macros: window.KATEX_MACROS || {}, trust: true,
    strict: false, throwOnError: false});
  else node.textContent = tex;
}

/* ---------- colour maps -------------------------------------------- */
function lut(stops){
  const rgb = stops.map(s => [1,3,5].map(i => parseInt(s.slice(i, i+2), 16)));
  const out = [];
  for(let i = 0; i < 256; i++){
    const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
    const c = rgb[k].map((v, j) => Math.round(v + (rgb[k+1][j] - v) * f));
    out.push('rgb(' + c.join(',') + ')');
  }
  return out;
}
/* viridis: perceptually uniform and colour-blind safe */
const VIRIDIS = lut(['#440154','#482475','#414487','#355f8d','#2a788e','#21918c',
                     '#22a884','#44bf70','#7ad151','#bddf26','#fde725']);
const cmap = t => VIRIDIS[clamp(Math.round(t * 255), 0, 255)];
/* grey, for a field shown on its own range rather than a shared one */
const GREY = lut(['#4a585e', '#9ba3a1', '#e4dfcd']);
const gmap = t => GREY[clamp(Math.round(t * 255), 0, 255)];

/* the end labels sit above and below the bar, where a long one cannot reach
   the field; map is the colour map (viridis by default), label formats an end */
function colourbar(c, x, y, w, hgt, lo, hi, logscale, P, map, label){
  map = map || cmap;
  for(let i = 0; i < hgt; i++){
    c.fillStyle = map(1 - i / (hgt - 1));
    c.fillRect(x, y + i, w, 1.2);
  }
  c.strokeStyle = P.rule2; c.lineWidth = 1; c.strokeRect(x, y, w, hgt);
  const lab = label || (v => logscale ? dec(Math.round(Math.log10(v))) : sci(v));
  text(c, x + w, y - 7, lab(hi), P.muted, 10.5, 'right');
  text(c, x + w, y + hgt + 8, lab(lo), P.muted, 10.5, 'right');
  if(logscale){
    const e0 = Math.log10(lo), e1 = Math.log10(hi);
    for(let e = Math.ceil(e0); e <= Math.floor(e1); e++){
      if((e - Math.ceil(e0)) % 4 || e === Math.round(e0) || e === Math.round(e1)) continue;
      const yy = y + hgt * (1 - (e - e0) / (e1 - e0));
      text(c, x - 4, yy, dec(e), P.faint, 10, 'right');
    }
  }
}

/* ---------- meshes --------------------------------------------------- */
function drawField(c, mesh, vals, x0, y0, S0, colourOf){
  const xy = mesh.xy, tri = mesh.tri;
  c.lineWidth = 0.6; c.lineJoin = 'round';
  for(let k = 0; k < mesh.nc; k++){
    const a = tri[3*k], b = tri[3*k+1], d = tri[3*k+2];
    const col = colourOf(vals[k]);
    c.fillStyle = col; c.strokeStyle = col;
    c.beginPath();
    c.moveTo(x0 + xy[2*a]*S0, y0 + (1 - xy[2*a+1])*S0);
    c.lineTo(x0 + xy[2*b]*S0, y0 + (1 - xy[2*b+1])*S0);
    c.lineTo(x0 + xy[2*d]*S0, y0 + (1 - xy[2*d+1])*S0);
    c.closePath(); c.fill(); c.stroke();
  }
}
/* the mesh itself, faintly, over a field: one path, so shared edges do not darken */
function drawEdges(c, mesh, x0, y0, S0, col, alpha){
  const xy = mesh.xy, tri = mesh.tri;
  c.save();
  c.globalAlpha = alpha; c.strokeStyle = col; c.lineWidth = 0.5; c.lineJoin = 'round';
  c.beginPath();
  for(let k = 0; k < mesh.nc; k++){
    const a = tri[3*k], b = tri[3*k+1], d = tri[3*k+2];
    c.moveTo(x0 + xy[2*a]*S0, y0 + (1 - xy[2*a+1])*S0);
    c.lineTo(x0 + xy[2*b]*S0, y0 + (1 - xy[2*b+1])*S0);
    c.lineTo(x0 + xy[2*d]*S0, y0 + (1 - xy[2*d+1])*S0);
    c.closePath();
  }
  c.stroke();
  c.restore();
}

/* ==================================================================== *
 *  1. Convergence plots                                                *
 * ==================================================================== */
const DEFAULT_YS = 'sigma_error|‖σ − σₕ‖ in H(div);displacement_error|‖u − uₕ‖ in L²;omega_err|‖ω − ωₕ‖ in L²';
const MARK = ['circle', 'triangle', 'square'];
function marker(g, kind, x, y, r, col){
  if(kind === 'circle') return S('circle', {cx:x, cy:y, r:r, fill:col}, g);
  if(kind === 'triangle') return S('path', {d:`M${x},${y-r*1.25}L${x+r*1.15},${y+r*.8}L${x-r*1.15},${y+r*.8}Z`, fill:col}, g);
  return S('rect', {x:x-r*.9, y:y-r*.9, width:r*1.8, height:r*1.8, fill:col}, g);
}
function paperSeries(name, param, v, col){
  const D = (window.PAPER || {})[name] || ((window.TORQUE || {}).tables || {})[name];
  if(!D) return null;
  const ip = D.cols.indexOf(param), ir = D.cols.indexOf('ref'), iy = D.cols.indexOf(col);
  if(iy < 0) return null;
  return D.rows.filter(r => Math.abs(r[ip] - v) < 1e-6 * v).map(r => [r[ir], r[iy]])
    .sort((a, b) => a[0] - b[0]);
}

function Plot(host){
  const kind = host.dataset.kind || 'elasticity';
  // the paper's delta and Ra; L, the symmetric load of the torque slides
  const param = kind === 'stokes' ? 'Ra' : kind === 'torque' ? 'L' : 'Bnd';
  const plab = kind === 'stokes' ? 'Ra' : kind === 'torque' ? 'L' : 'δ';
  const panels = host.dataset.panels.split(';').map(s => {
    const [f, t] = s.split('|'); return {file: f.trim(), title: (t || f).trim()}; });
  const ys = (host.dataset.ys || DEFAULT_YS).split(';').map(s => {
    const [c, l] = s.split('|'); return {col: c.trim(), label: l.trim()}; });
  const ylim0 = host.dataset.ylim ? host.dataset.ylim.split(',').map(Number) : null;
  const H0 = +(host.dataset.h || 380);
  const vals = [10, 1000, 100000];
  let ycol = ys[0].col;
  const hidden = new Set();

  host.classList.add('wd', 'plot');
  host.innerHTML = '';
  const bar = h('div', 'bar');
  host.appendChild(bar);
  let ysel = null;
  if(ys.length > 1){
    bar.appendChild(h('span', 'ctl', '<span class="k">error shown</span>'));
    ysel = seg(ys.map(y => [y.col, y.label]), k => { ycol = k; ysel.set(k); draw(); });
    ysel.set(ycol);
    bar.appendChild(ysel);
  }
  const pw = h('div', 'panels'); host.appendChild(pw);
  const legend = h('div', 'legend'); host.appendChild(legend);
  const tip = h('div', 'tip'); host.appendChild(tip);

  function buildLegend(P){
    legend.innerHTML = '';
    vals.forEach((v, i) => {
      const it = h('span', hidden.has(v) ? 'off' : '');
      const sv = S('svg', {width:26, height:14});
      S('line', {x1:0, y1:7, x2:26, y2:7, stroke:P.d[i], 'stroke-width':2}, sv);
      marker(sv, MARK[i], 13, 7, 4, P.d[i]);
      it.appendChild(sv);
      it.appendChild(document.createTextNode(`${plab} = ${v === 10 ? '10' : dec(Math.round(Math.log10(v)))}`));
      it.addEventListener('click', e => { e.stopPropagation();
        hidden.has(v) ? hidden.delete(v) : hidden.add(v); draw(); });
      legend.appendChild(it);
    });
  }

  function draw(){
    const P = pal();
    buildLegend(P);
    pw.innerHTML = '';
    const W = host.clientWidth || 1212;
    const H = Math.max(H0, (host.clientHeight || 0) - bar.offsetHeight - legend.offsetHeight - 16);
    const N = panels.length, gap = 14, L0 = 70, L1 = 18, R = 12, T = 30, B = 44;
    const w = (W - gap*(N-1) - L0 - L1*(N-1) - R*N) / N;
    pw.style.gridTemplateColumns = panels.map((_, i) => (w + (i ? L1 : L0) + R) + 'px').join(' ');
    const ph = H - T - B;
    // data and ranges
    let maxRef = 0, lo = Infinity, hi = 0;
    const data = panels.map(p => vals.map(v => {
      const s = paperSeries(p.file, param, v, ycol) || [];
      s.forEach(([r, y]) => { maxRef = Math.max(maxRef, r);
        if(y > 0 && !hidden.has(v)){ lo = Math.min(lo, y); hi = Math.max(hi, y); } });
      return s;
    }));
    let e0, e1;
    if(ylim0 && ycol === ys[0].col){ e0 = Math.log10(ylim0[0]); e1 = Math.log10(ylim0[1]); }
    else if(hi > 0){ e0 = Math.floor(Math.log10(lo)); e1 = Math.ceil(Math.log10(hi));
      if(e1 - e0 < 2){ e0 -= 1; e1 += 1; } }
    else { e0 = -16; e1 = 0; }
    const every = (e1 - e0) > 10 ? 3 : (e1 - e0) > 7 ? 2 : 1;
    panels.forEach((p, pi) => {
      const L = pi ? L1 : L0;
      const svg = S('svg', {width: L + w + R, height: H});
      pw.appendChild(svg);
      const X = r => L + (r + 0.25) / (maxRef + 0.5) * w;
      const Y = y => T + (e1 - Math.log10(y)) / (e1 - e0) * ph;
      S('rect', {x:L, y:T, width:w, height:ph, fill:'none', stroke:P.rule2}, svg);
      for(let e = Math.ceil(e0); e <= Math.floor(e1); e++){
        const y = Y(Math.pow(10, e));
        S('line', {x1:L, x2:L+w, y1:y, y2:y, stroke:P.rule, 'stroke-dasharray':'3 3'}, svg);
        if(pi === 0 && (e - Math.ceil(e0)) % every === 0){
          const t = S('text', {x:L-6, y:y+4, 'text-anchor':'end', 'font-size':11.5, fill:P.muted}, svg);
          t.textContent = dec(e);
        }
      }
      for(let r = 0; r <= maxRef; r++){
        S('line', {x1:X(r), x2:X(r), y1:T+ph, y2:T+ph+4, stroke:P.rule2}, svg);
        const t = S('text', {x:X(r), y:T+ph+17, 'text-anchor':'middle', 'font-size':11.5, fill:P.muted}, svg);
        t.textContent = r;
      }
      const xl = S('text', {x:L+w/2, y:H-8, 'text-anchor':'middle', 'font-size':12, fill:P.ink2}, svg);
      xl.textContent = 'refinement level';
      const tt = S('text', {x:L+w/2, y:T-10, 'text-anchor':'middle', 'font-size':14, fill:P.ink, 'font-weight':600}, svg);
      tt.textContent = p.title;
      if(pi === 0){
        const yl = S('text', {x:14, y:T+ph/2, 'text-anchor':'middle', 'font-size':12, fill:P.ink2,
          transform:`rotate(-90 14 ${T+ph/2})`}, svg);
        yl.textContent = ys.find(y => y.col === ycol).label;
      }
      const any = data[pi].some(s => s.some(([, y]) => y > 0));
      if(!any){
        const m = S('text', {x:L+w/2, y:T+ph/2, 'text-anchor':'middle', 'font-size':13, fill:P.faint,
          'font-style':'italic'}, svg);
        m.textContent = ycol === 'omega_err' ? 'no multiplier: symmetry is built into Σₕ' : 'no data';
        return;
      }
      const clip = 'clip' + Math.random().toString(36).slice(2);
      const cp = S('clipPath', {id:clip}, S('defs', {}, svg));
      S('rect', {x:L, y:T-6, width:w, height:ph+12}, cp);
      data[pi].forEach((s, i) => {
        if(hidden.has(vals[i])) return;
        const pts = s.filter(([, y]) => y > 0);
        if(!pts.length) return;
        const g = S('g', {'clip-path':`url(#${clip})`}, svg);
        S('polyline', {points: pts.map(([r, y]) => X(r) + ',' + Y(y)).join(' '),
          fill:'none', stroke:P.d[i], 'stroke-width':2.1}, g);
        pts.forEach(([r, y]) => {
          const m = marker(g, MARK[i], X(r), Y(y), 4.4, P.d[i]);
          m.style.cursor = 'crosshair';
          m.addEventListener('mouseenter', e => {
            tip.innerHTML = `<b>${p.title}</b><br>${plab} = ${vals[i] === 10 ? '10' : dec(Math.round(Math.log10(vals[i])))}, level ${r}<br>${sci(y, 3)}`;
            const [x0, y0] = local(e, host);
            tip.style.left = (x0 + 12) + 'px'; tip.style.top = (y0 - 10) + 'px'; tip.style.display = 'block';
          });
          m.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
        });
      });
    });
  }
  return {draw};
}

/* ==================================================================== *
 *  2. Robustness matrix                                                *
 * ==================================================================== */
const MX_COLS = [['rigid_body_motion', 'Rigid body', 'isotropic'], ['transverse_isotropic', 'Transverse', 'isotropy'],
                 ['polar', 'Polar fluid', '2D'], ['polar_3d', 'Polar fluid', '3D'],
                 ['polar_extra', 'Polar + stress', 'σ ≠ 0']];
const MX_ROWS = [['peers_1', 'PEERS₁', 'weak'], ['afw_1', 'AFW₁', 'weak'], ['afw_2', 'AFW₂', 'weak'],
                 ['afw_3', 'AFW₃', 'weak'], ['jm_1', 'JMK', 'strong'], ['hz_3', 'HZ₃', 'strong']];
const MX_TOL = 1e-10;
/* what the theory predicts for a run the paper did not make, and why:
   strong symmetry is always robust, weak symmetry exactly when Xi_h contains omega* */
function expected(pre, suf, sym){
  if(sym === 'strong') return [true, 'it is strongly symmetric, so its error estimate has no ω term'];
  if(pre === 'rigid_body_motion') return [true, 'ω* is constant, and every Ξₕ contains the constants'];
  if(pre === 'transverse_isotropic') return suf === 'afw_3' ? [true, 'ω* ∈ 𝒫₂ = Ξₕ'] : [false, 'ω* ∈ 𝒫₂ is not in Ξₕ'];
  return [false, 'ω* is not a polynomial, so no Ξₕ contains it'];
}

function Matrix(host){
  host.classList.add('wd', 'matrix');
  host.innerHTML = '';
  const wrapg = h('div', null); wrapg.style.cssText = 'display:grid;grid-template-columns:auto 1fr;gap:26px;height:100%';
  host.appendChild(wrapg);
  const left = h('div'); wrapg.appendChild(left);
  const right = h('div'); right.style.cssText = 'display:flex;flex-direction:column;gap:6px;min-width:0';
  wrapg.appendChild(right);
  const tip = h('div', 'tip'); host.appendChild(tip);
  const showTip = (e, html) => {
    tip.innerHTML = html;
    const [x, y] = local(e, host);
    tip.style.left = (x + 14) + 'px'; tip.style.top = (y + 8) + 'px'; tip.style.display = 'block';
  };

  function classify(name){
    const D = (window.PAPER || {})[name];
    if(!D) return null;
    const ib = D.cols.indexOf('Bnd'), ir = D.cols.indexOf('ref'), ie = D.cols.indexOf('sigma_error');
    const L = Math.max(...D.rows.map(r => r[ir]));
    const at = v => D.rows.find(r => r[ir] === L && Math.abs(r[ib] - v) < 1e-6 * v)[ie];
    const e10 = at(10), e5 = at(1e5);
    const g = (e5 - e10) / (1e5 - 10);
    return {L, e10, e5, g, ok: g < MX_TOL};
  }
  let sel = ['polar', 'afw_3'];
  const tbl = h('table');
  const thead = h('tr');
  thead.appendChild(h('th'));
  MX_COLS.forEach(([, a, b]) => thead.appendChild(h('th', null, a + '<br><span style="text-transform:none;letter-spacing:0">' + b + '</span>')));
  tbl.appendChild(thead);
  const cells = [];
  let group = '';
  MX_ROWS.forEach(([suf, lab, sym]) => {
    if(sym !== group){
      group = sym;
      const gr = h('tr', 'grp');
      const gt = h('td', null, `<span class="ctl"><span class="k" style="color:var(--${sym})">${sym} symmetry</span></span>`);
      gt.colSpan = 6; gt.style.textAlign = 'left'; gt.style.paddingLeft = '8px';
      gr.appendChild(gt); tbl.appendChild(gr);
    }
    const tr = h('tr');
    tr.appendChild(h('th', 'row', lab));
    MX_COLS.forEach(([pre]) => {
      const name = pre + '_' + suf, r = classify(name), ex = r ? null : expected(pre, suf, sym);
      const td = h('td', r ? (r.ok ? 'ok' : 'bad') : (ex[0] ? 'ok' : 'bad') + ' exp');
      if(r){
        td.innerHTML = `<span class="g">${r.ok ? '✓' : '✗'}</span><span class="v">${sci(r.e5)}</span>`;
        td.addEventListener('mouseenter', e => showTip(e, `<b>${lab}, ${pre.replace(/_/g, ' ')}</b><br>finest level ${r.L}<br>` +
          `δ = 10: ${sci(r.e10, 3)}<br>δ = 10⁵: ${sci(r.e5, 3)}<br>growth per unit δ: ${sci(r.g, 2)}`));
        td.addEventListener('click', e => { e.stopPropagation(); sel = [pre, suf]; update(); });
      } else {
        td.innerHTML = `<span class="g">${ex[0] ? '✓' : '✗'}<sup>*</sup></span><span class="v">not run</span>`;
        td.addEventListener('mouseenter', e => showTip(e, `<b>${lab}, ${pre.replace(/_/g, ' ')}</b><br>not run, but we expect it to be ` +
          `${ex[0] ? 'material robust' : 'not material robust'}:<br>${ex[1]}`));
      }
      td.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
      td.dataset.k = pre + '|' + suf;
      cells.push(td);
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
  left.appendChild(tbl);
  left.appendChild(h('p', 'tiny dim',
    `Each cell is the stress error on the finest mesh at δ = 10⁵. ✓ means it grows by less than ${dec(-10)} per unit δ (roundoff), ` +
    `✗ that the scheme feels the stress-free state. Cells marked * were not run and show what the theory predicts: strong symmetry ` +
    `is always robust, and weak symmetry is robust exactly when Ξₕ contains ω*.`));
  left.lastChild.style.cssText = 'max-width:700px;margin-top:8px;font-family:var(--serif)';

  const cap = h('div', 'small'); right.appendChild(cap);
  const plotHost = h('div'); right.appendChild(plotHost);
  let plot = null;
  function update(){
    cells.forEach(td => td.classList.toggle('sel', td.dataset.k === sel.join('|')));
    const [pre, suf] = sel, name = pre + '_' + suf, r = classify(name);
    const row = MX_ROWS.find(x => x[0] === suf), col = MX_COLS.find(x => x[0] === pre);
    cap.innerHTML = `<b>${row[1]}</b> on <b>${col[1]} (${col[2]})</b>: ` + (r.ok
      ? '<b class="cstrong">material robust</b>. The error does not grow with δ beyond roundoff.'
      : `<b class="alert">not material robust</b>. The error grows like δ, by ${sci(r.g)} per unit δ.`);
    plotHost.dataset.panels = name + '|' + row[1] + ', ' + col[1] + ' ' + col[2];
    plotHost.dataset.h = 330;
    plot = Plot(plotHost);
    plot.draw();
  }
  function draw(){ update(); }
  return {draw};
}

/* ==================================================================== *
 *  3. The stress-free state lab                                        *
 * ==================================================================== */
const EXLAB = {rigid: 'Rigid body', trans: 'Transverse', polar: 'Polar', extra: 'Polar + stress'};
const SCH = {peers1: 'PEERS₁', afw1: 'AFW₁', afw2: 'AFW₂', afw3: 'AFW₃', jm1: 'JMK', hz3: 'HZ₃'};
const NOTE = {
  rigid: 'Rigid body motions are the stress-free states of an isotropic solid. Here ω* is constant, every Ξₕ contains the constants, and both schemes stay at roundoff.',
  trans: 'Here ω* = (x² + xy + y²)/(2μ) is quadratic. AFW₃, with Ξₕ = 𝒫₂, captures it exactly. AFW₁, AFW₂ and PEERS₁ do not, and their stress grows like δ.',
  polar: 'Here ω* = −cos x sinh y / μ is not a polynomial, so every weakly symmetric scheme gets it wrong, and a higher k only changes the constant.',
  extra: 'Now σ ≠ 0 and does not depend on δ. The strong error stays at the discretisation error, while the weak error grows like δ once the kernel part takes over.',
};
/* the stress-free displacement w* (unit delta), for panel A */
function wstar(ex, mu){
  if(ex === 'rigid') return (x, y) => [-y, x];
  if(ex === 'trans') return (x, y) => [-(x**3 - y**3) / 3 / (2*mu),
                                        -(x*x*y + x*y*y + y**3/3 + 2*x**3/3) / (2*mu)];
  return (x, y) => [Math.cos(x)*Math.cosh(y) / mu, -Math.sin(x)*Math.sinh(y) / mu];
}
const quad = (G, d) => Math.max(0, G[0] + 2*d*G[1] + d*d*G[2]);
/* max over the unit square of the scalar rotation |omega*| of the unit stress-free state */
const OMAX = {rigid: () => 1, trans: m => 3 / (2*m), polar: m => Math.sinh(1) / m, extra: m => Math.sinh(1) / m};

function Lab(host){
  const LD = window.LAB;
  host.classList.add('wd', 'lab-w');
  if(!LD){ host.textContent = 'lab data missing: run make data'; return {draw(){}}; }
  const mode = host.dataset.mode || 'compact';
  const st = {ex: host.dataset.preset || 'polar', weak: host.dataset.weak || 'afw1', strong: 'jm1',
              lev: 2, view: 'err', ld: 3, hot: null};
  if(mode === 'full'){
    const m = /[?&]lab=([^&#]+)/.exec(location.search);
    if(m){
      const [ex, wk, sg, lv, d] = decodeURIComponent(m[1]).split(',');
      if(ex in EXLAB) st.ex = ex;
      if(wk in SCH) st.weak = wk;
      if(sg in SCH) st.strong = sg;
      if(lv) st.lev = clamp(+lv, 0, LD.levels);
      if(d) st.ld = clamp(Math.log10(+d), 0, 6);
    }
  }
  const meshes = LD.meshes.map(m => ({nv: m.nv, nc: m.nc, xy: f32(m.xy), tri: u32(m.cells)}));
  const FC = {};
  function fields(ex, sch, lev){
    const k = ex + ':' + sch + ':' + lev;
    if(FC[k]) return FC[k];
    const e = LD.cases[ex + ':' + sch].levels[lev];
    const F = {};
    for(const name in e.fields) F[name] = e.fields[name].map(f32);
    return FC[k] = F;
  }
  const rec = (sch, lev) => LD.cases[st.ex + ':' + sch].levels[lev];
  const mu = () => LD.mu[st.ex];
  const delta = () => Math.pow(10, st.ld);

  /* ---- DOM -------------------------------------------------------- */
  host.innerHTML = '';
  const full = mode === 'full';
  const top = h('div', 'row'); host.appendChild(top);
  const mkPane = (title) => {
    const p = h('div', 'pane'); const cv = h('canvas'); p.appendChild(cv);
    const t = h('div', 'ttl', title); p.appendChild(t);
    const big = h('div', 'big'); p.appendChild(big);
    const rd = h('div', 'rdo'); p.appendChild(rd);
    top.appendChild(p);
    return {p, cv, t, big, rd};
  };
  const A = mkPane('stress-free state  δ·w*');
  const Bs = mkPane('');
  const Bw = mkPane('');
  const C = full ? mkPane('') : null;
  if(C) C.cv.style.cursor = 'crosshair';

  const bottom = h('div'); host.appendChild(bottom);
  bottom.style.cssText = full ? 'display:grid;grid-template-columns:560px 1fr;gap:16px;margin-top:12px'
                              : 'margin-top:12px;display:flex;flex-direction:column;gap:8px';
  const ctl = h('div', 'stack'); ctl.style.gap = '8px'; bottom.appendChild(ctl);

  const ds = h('div', 'dslider');
  ds.innerHTML = '<span class="ctl"><span class="k">excite</span></span>';
  const rng = h('input'); rng.type = 'range'; rng.min = 0; rng.max = 6; rng.step = 0.01; rng.value = st.ld;
  const dv = h('span', 'dval');
  rng.addEventListener('input', () => { st.ld = +rng.value; draw(); });
  ds.appendChild(rng); ds.appendChild(dv);
  ctl.appendChild(ds);

  const line = (k, node) => { const r = h('div', 'ctl'); r.appendChild(h('span', 'k', k)); r.appendChild(node); ctl.appendChild(r); return r; };
  let exSeg = null, sSeg = null, lvSeg = null;
  const lvNote = h('span', 'tiny dim');
  if(full){
    exSeg = seg(Object.entries(EXLAB), k => { st.ex = k; draw(); });
    line('example', exSeg);
    sSeg = seg([['jm1', 'JMK'], ['hz3', 'HZ₃']], k => { st.strong = k; draw(); }, 'strongseg');
    line('strong', sSeg);
  }
  const wSeg = seg([['peers1', 'PEERS₁'], ['afw1', 'AFW₁'], ['afw2', 'AFW₂'], ['afw3', 'AFW₃']],
                   k => { st.weak = k; draw(); }, 'weakseg');
  line('weak', wSeg);
  if(full){
    lvSeg = seg([0,1,2,3].map(l => [String(l), 'level ' + l]), k => { st.lev = +k; draw(); });
    line('mesh', lvSeg).appendChild(lvNote);
  }
  const vSeg = seg([['err', 'stress error |σ − σₕ|'], ['skw', 'torque |skw σₕ|'], ['poll', 'μ |ω − Π<sub>Ξ</sub>ω|']],
                   k => { st.view = k; draw(); });
  line('show', vSeg);
  const note = h('div', 'tiny'); note.style.cssText = 'color:var(--ink2);font-family:var(--serif);max-width:' + (full ? '560px' : '1100px');
  ctl.appendChild(note);

  let est = null, estS = null, estW = null, numS = null, numW = null;
  if(full){
    est = h('div', 'card est');
    est.innerHTML = '<div class="lab">The two estimates, evaluated</div>';
    const rowS = h('div'); estS = h('div'); numS = h('div', 'row2');
    rowS.appendChild(estS); rowS.appendChild(numS);
    const rowW = h('div'); rowW.style.marginTop = '8px'; estW = h('div'); numW = h('div', 'row2');
    rowW.appendChild(estW); rowW.appendChild(numW);
    est.appendChild(rowS); est.appendChild(rowW);
    const hint = h('div', 'tiny dim', 'Hover over a coloured term to find its curve. C is the ratio of the two sides. Wherever the pollution term is above roundoff, C stays between 0.75 and 2.2 for every example, weak scheme, mesh and δ, so the estimate is sharp.');
    hint.style.cssText = 'margin-top:10px;font-family:var(--serif)';
    est.appendChild(hint);
    bottom.appendChild(est);
    katexInto(estS, '\\displaystyle\\htmlClass{cstrong}{\\text{strong: }}\\norm{\\tensor{\\sigma}-\\tensor{\\sigma}_h}_{\\div}\\leq C\\,' +
      '\\htmlClass{term capprox tapprox}{\\inf_{\\tensor{\\tau}_h\\in\\Sigma_h^{\\sym}}\\norm{\\tensor{\\sigma}-\\tensor{\\tau}_h}_{\\div}}');
    katexInto(estW, '\\displaystyle\\htmlClass{cweak}{\\text{weak: }}\\norm{\\tensor{\\sigma}-\\tensor{\\sigma}_h}_{\\div}\\leq C\\Big(' +
      '\\htmlClass{term capprox tapprox}{\\inf_{\\tensor{\\tau}_h\\in\\Sigma_h}\\norm{\\tensor{\\sigma}-\\tensor{\\tau}_h}_{\\div}}+' +
      '\\htmlClass{term cpoll tpoll}{\\mu\\inf_{\\tensor{\\zeta}_h\\in\\Xi_h}\\norm{\\tensor{\\omega}-\\tensor{\\zeta}_h}}\\Big)');
    est.querySelectorAll('.term').forEach(t => {
      const which = t.classList.contains('tpoll') ? 'poll' : 'approx';
      t.addEventListener('mouseenter', () => { st.hot = which; t.classList.add('hot'); drawC(); });
      t.addEventListener('mouseleave', () => { st.hot = null; t.classList.remove('hot'); drawC(); });
    });
  }

  /* ---- layout ------------------------------------------------------ */
  let P0 = 0, Ph = 0, Cw = 0;
  function layout(){
    const W = host.clientWidth || 1212, H = host.clientHeight || (full ? 610 : 600);
    if(full){
      P0 = 250; Ph = 290;
      Cw = W - 3*P0 - 3*12;
      top.style.gridTemplateColumns = `${P0}px ${P0}px ${P0}px ${Cw}px`;
    } else {
      P0 = Math.floor((W - 2*14) / 3); Ph = Math.min(P0, H - 150);
      top.style.gridTemplateColumns = `repeat(3, ${P0}px)`;
    }
    top.style.gap = full ? '12px' : '14px';
    [A, Bs, Bw].forEach(p => { p.p.style.height = Ph + 'px'; });
    if(C) C.p.style.height = Ph + 'px';
  }

  /* ---- panel A: the excitation ---------------------------------- */
  let raf = null, t0 = performance.now();
  function drawA(){
    const P = pal(), w = P0, hh = Ph;
    const c = sizeCanvas(A.cv, w, hh);
    const Sq = Math.min(w, hh) - 64, x0 = (w - Sq) / 2, y0 = (hh - Sq) / 2 + 4;
    const f = wstar(st.ex, mu());
    let wmax = 0;
    for(let i = 0; i <= 20; i++) for(let j = 0; j <= 20; j++){
      const v = f(i/20, j/20); wmax = Math.max(wmax, Math.hypot(v[0], v[1])); }
    const t = (performance.now() - t0) / 1000;
    const amp = 0.16 * (st.ld / 6) * (0.55 + 0.45 * Math.sin(2.2 * t));
    const mesh = meshes[1], xy = mesh.xy, tri = mesh.tri;
    const map = (x, y) => { const v = f(x, y); return [x + amp * v[0] / wmax, y + amp * v[1] / wmax]; };
    // undeformed outline
    c.strokeStyle = P.rule2; c.setLineDash([4, 4]); c.lineWidth = 1;
    c.strokeRect(x0, y0, Sq, Sq); c.setLineDash([]);
    // deformed mesh
    c.strokeStyle = P.faint; c.lineWidth = 0.6;
    c.fillStyle = P.paper;
    const X = p => x0 + p[0]*Sq, Y = p => y0 + (1 - p[1])*Sq;
    for(let k = 0; k < mesh.nc; k++){
      const q = [tri[3*k], tri[3*k+1], tri[3*k+2]].map(v => map(xy[2*v], xy[2*v+1]));
      c.beginPath(); c.moveTo(X(q[0]), Y(q[0])); c.lineTo(X(q[1]), Y(q[1])); c.lineTo(X(q[2]), Y(q[2]));
      c.closePath(); c.fill(); c.stroke();
    }
    // arrows of w*
    c.strokeStyle = P.accent; c.fillStyle = P.accent; c.lineWidth = 1.4;
    for(let i = 1; i < 8; i++) for(let j = 1; j < 8; j++){
      const x = i/8, y = j/8, v = f(x, y), n = Math.hypot(v[0], v[1]) / wmax;
      if(n < 1e-3) continue;
      const L = 0.055 * Math.sqrt(n), a = Math.atan2(v[1], v[0]);
      const p = map(x, y), px = X(p), py = Y(p);
      const ex = px + Math.cos(a) * L * Sq, ey = py - Math.sin(a) * L * Sq;
      c.beginPath(); c.moveTo(px, py); c.lineTo(ex, ey); c.stroke();
      c.beginPath(); c.moveTo(ex, ey);
      c.lineTo(ex - Math.cos(a - .45)*6, ey + Math.sin(a - .45)*6);
      c.lineTo(ex - Math.cos(a + .45)*6, ey + Math.sin(a + .45)*6); c.closePath(); c.fill();
    }
    A.big.innerHTML = st.ex === 'extra' ? '<span style="color:var(--approx)">σ ≠ 0, fixed</span>'
                                        : '<span class="cstrong">σ ≡ 0</span>';
    let n2 = 0;
    for(let i = 0; i < 40; i++) for(let j = 0; j < 40; j++){ const v = f((i+.5)/40, (j+.5)/40); n2 += (v[0]*v[0] + v[1]*v[1]) / 1600; }
    A.rd.textContent = '‖δw*‖ = ' + sci(delta() * Math.sqrt(n2));
  }
  function loop(){
    // a hidden slide (display: none) has no offsetParent: stop animating it
    if(!host.offsetParent){ raf = null; return; }
    drawA();
    raf = requestAnimationFrame(loop);
  }

  /* ---- panels B: the discrete stress ----------------------------- */
  // ro: the whole field is roundoff. Every cell's response to the unit
  // stress-free state is below RO of that state's size in stress units,
  // mu * max|omega*|, and its delta-independent part (extra only, whose exact
  // stress is of order one) below RO0. The lab's cells sit three decades or
  // more to either side of both
  const RO = 1e-9, RO0 = 1e-12;
  function paneValues(sch, lev){
    const F = fields(st.ex, sch, lev), d = delta();
    const G = st.view === 'err' ? F.err : st.view === 'skw' ? F.skw : F.poll;
    if(!G) return null;
    const n = G[2].length, v = new Float64Array(n), m = st.view === 'poll' ? mu() : 1;
    const c1 = (RO * mu() * OMAX[st.ex](mu()) / m) ** 2, c0 = (RO0 / m) ** 2;
    let ro = true;
    for(let k = 0; k < n; k++){
      const a = G[0] ? G[0][k] : 0, b = G[1] ? G[1][k] : 0, cc = G[2][k];
      if(cc > c1 || a > c0) ro = false;
      v[k] = m * Math.sqrt(Math.max(0, a + 2*d*b + d*d*cc));
    }
    return {v, ro};
  }
  // what an empty panel means, in two centred lines
  function caption(c, x, y, lines, P){
    c.font = 'italic 14px ui-sans-serif,-apple-system,Helvetica,Arial,sans-serif';
    const w = Math.max(...lines.map(s => c.measureText(s).width)) + 24, hh = 20 * lines.length + 14;
    c.fillStyle = P.paper; c.strokeStyle = P.rule2; c.lineWidth = 1;
    c.beginPath(); c.roundRect(x - w/2, y - hh/2, w, hh, 6); c.fill(); c.stroke();
    lines.forEach((s, i) => text(c, x, y + (i - (lines.length - 1) / 2) * 20, s, P.muted, 14, 'center', 'middle', 'italic '));
  }
  const BUILTIN = {skw: ['skw σₕ ≡ 0:', 'symmetry is built into Σₕ'], poll: ['no multiplier:', 'symmetry is built into Σₕ']};
  function norms(sch, lev){
    const e = rec(sch, lev), d = delta();
    const r = {err: Math.sqrt(quad(e.sigma, d)), skw: Math.sqrt(quad(e.skw, d))};
    if(e.poll) r.poll = mu() * Math.sqrt(quad(e.poll, d));
    r.best = e.best0;
    return r;
  }
  function drawB(){
    const P = pal(), lev = Math.min(st.lev, LD.field_levels), mesh = meshes[lev];
    [[Bs, st.strong, 'strong'], [Bw, st.weak, 'weak']].forEach(([pn, sch, sym]) => {
      const w = P0, hh = Ph, pv = paneValues(sch, lev);
      const c = sizeCanvas(pn.cv, w, hh);
      const Sq = Math.min(w - 56, hh - 72), x0 = (w - 40 - Sq) / 2, y0 = (hh - Sq) / 2 + 2;
      // every field on its own log range, so its pattern over the mesh shows (on one
      // range for all fields, a scheme's error varied by 2 decades out of 14 and
      // looked flat). Roundoff in grey, so it cannot pass for a real error, which
      // is in colour; the size is in the colour bar and the norm below
      let flo = Infinity, fhi = 0;
      if(pv) for(const x of pv.v) if(x > 0){ flo = Math.min(flo, x); fhi = Math.max(fhi, x); }
      if(!(fhi > 0)){
        // nothing to draw: no multiplier, or the skew part of a symmetric stress
        c.fillStyle = P.paper; c.fillRect(x0, y0, Sq, Sq);
        c.strokeStyle = P.rule2; c.lineWidth = 1; c.strokeRect(x0, y0, Sq, Sq);
        caption(c, x0 + Sq/2, y0 + Sq/2, !pv ? BUILTIN.poll
          : sym === 'strong' && st.view === 'skw' ? BUILTIN.skw : ['identically zero'], P);
      } else {
        flo = Math.max(flo, fhi * 1e-6);
        const span = Math.max(Math.log10(fhi / flo), 1e-6), map = pv.ro ? gmap : cmap;
        drawField(c, mesh, pv.v, x0, y0, Sq, x => map(x <= flo ? 0 : Math.log10(x / flo) / span));
        drawEdges(c, mesh, x0, y0, Sq, P.ink, 0.13);
        c.strokeStyle = P.rule2; c.lineWidth = 1; c.strokeRect(x0, y0, Sq, Sq);
        colourbar(c, w - 22, y0, 11, Sq, flo, fhi, true, P, map, sci);
      }
      const own = pv && pv.ro && fhi > 0;
      pn.t.innerHTML = `<span style="color:var(--${sym})">${SCH[sch]} · ${sym} symmetry</span>` +
        (own ? '<span class="ro">roundoff</span>' : '');
      const nm = norms(sch, st.lev);
      const val = st.view === 'err' ? nm.err : st.view === 'skw' ? nm.skw : nm.poll;
      const lab = st.view === 'err' ? '‖σ − σₕ‖<sub>div</sub>' : st.view === 'skw' ? '‖skw σₕ‖' : 'μ‖ω − Πω‖';
      pn.big.innerHTML = val == null ? '' : `<span style="color:var(--${sym})">${lab} = ${sci(val)}</span>`;
      // the full lab names the mesh in its controls; the compact one here.
      // Cell fields were stored up to FIELD_LEVELS, so say when a coarser one shows
      pn.rd.textContent = full ? '' : 'mesh level ' + lev;
      pn.rd.classList.add('tr');
    });
  }

  /* ---- panel C: error against delta -------------------------------- */
  function drawC(){
    if(!C) return;
    const P = pal(), w = Cw, hh = Ph;
    const c = sizeCanvas(C.cv, w, hh);
    const L = 58, R = 14, T = 30, B = 40, pw_ = w - L - R, ph = hh - T - B;
    const es = rec(st.strong, st.lev), ew = rec(st.weak, st.lev), m = mu();
    const curves = [
      {k: 'strong', col: P.strong, f: d => Math.sqrt(quad(es.sigma, d)), w: 2.4},
      {k: 'weak', col: P.weak, f: d => Math.sqrt(quad(ew.sigma, d)), w: 2.4},
      {k: 'poll', col: P.poll, f: d => m * Math.sqrt(quad(ew.poll, d)), w: 1.8, dash: [6, 4]},
    ];
    if(ew.best0 > 0) curves.push({k: 'approx', col: P.approx, f: () => ew.best0, w: 1.8, dash: [2, 4]});
    let lo = Infinity, hi = 0;
    curves.forEach(cv => { for(let e = 0; e <= 6; e += 0.25){ const v = cv.f(Math.pow(10, e));
      if(v > 0){ lo = Math.min(lo, v); hi = Math.max(hi, v); } } });
    let e0 = Math.floor(Math.log10(lo)) - 1, e1 = Math.ceil(Math.log10(hi)) + 1;
    e0 = Math.max(e0, e1 - 22);
    const X = d => L + Math.log10(d) / 6 * pw_;
    const Y = v => T + (e1 - Math.log10(Math.max(v, Math.pow(10, e0)))) / (e1 - e0) * ph;
    // axes
    c.strokeStyle = P.rule; c.lineWidth = 1; c.setLineDash([3, 3]);
    const every = Math.ceil((e1 - e0) / 8);
    for(let e = e0; e <= e1; e++){
      if((e - e0) % every) continue;
      const y = Y(Math.pow(10, e));
      c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw_, y); c.stroke();
      text(c, L - 6, y, dec(e), P.muted, 10.5, 'right');
    }
    c.setLineDash([]);
    for(let e = 0; e <= 6; e++){
      const x = X(Math.pow(10, e));
      c.strokeStyle = P.rule2; c.beginPath(); c.moveTo(x, T + ph); c.lineTo(x, T + ph + 4); c.stroke();
      text(c, x, T + ph + 14, dec(e), P.muted, 10.5);
    }
    c.strokeStyle = P.rule2; c.strokeRect(L, T, pw_, ph);
    text(c, L + pw_/2, hh - 9, 'δ, the size of the stress-free state', P.ink2, 11.5);
    // curves
    curves.forEach(cv => {
      c.strokeStyle = cv.col; c.lineWidth = cv.w + (st.hot === cv.k ? 2 : 0);
      c.setLineDash(cv.dash || []);
      c.globalAlpha = st.hot && st.hot !== cv.k && (cv.k === 'poll' || cv.k === 'approx') ? 0.35 : 1;
      c.beginPath();
      for(let i = 0; i <= 240; i++){ const d = Math.pow(10, 6*i/240), x = X(d), y = Y(cv.f(d));
        i ? c.lineTo(x, y) : c.moveTo(x, y); }
      c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
    });
    // cursor
    const d = delta(), xc = X(d);
    c.strokeStyle = P.ink2; c.lineWidth = 1; c.setLineDash([2, 3]);
    c.beginPath(); c.moveTo(xc, T); c.lineTo(xc, T + ph); c.stroke(); c.setLineDash([]);
    curves.forEach(cv => { const v = cv.f(d); if(v > 0){ c.fillStyle = cv.col;
      c.beginPath(); c.arc(xc, Y(v), 3.6, 0, 2*Math.PI); c.fill(); } });
    // legend
    const items = [['strong ' + SCH[st.strong], P.strong, []], ['weak ' + SCH[st.weak], P.weak, []],
                   ['μ inf‖ω − ζₕ‖', P.poll, [6, 4]]];
    if(ew.best0 > 0) items.push(['inf‖σ − τₕ‖', P.approx, [2, 4]]);
    let lx = L + 6;
    items.forEach(([s, col, dash]) => {
      if(dash){ c.strokeStyle = col; c.lineWidth = 2; c.setLineDash(dash);
        c.beginPath(); c.moveTo(lx, 17); c.lineTo(lx + 16, 17); c.stroke(); c.setLineDash([]); lx += 20; }
      text(c, lx, 17, s, col, 11, 'left');
      c.font = '11px ui-sans-serif,-apple-system,Helvetica,Arial,sans-serif';
      lx += c.measureText(s).width + 12;
    });
  }

  function drawText(){
    dv.innerHTML = 'δ = ' + (st.ld < 0.005 ? '1' : sci(delta(), 2));
    note.textContent = NOTE[st.ex];
    wSeg.set(st.weak); vSeg.set(st.view);
    if(exSeg) exSeg.set(st.ex); if(sSeg) sSeg.set(st.strong); if(lvSeg) lvSeg.set(String(st.lev));
    lvNote.textContent = st.lev > LD.field_levels ? `heatmaps on level ${LD.field_levels}` : '';
    if(full){
      const ns = norms(st.strong, st.lev), nw = norms(st.weak, st.lev);
      // the ratio is only meaningful when the pollution term is above roundoff
      const above = nw.poll > 1e-9 * delta() * mu() * OMAX[st.ex](mu());
      const ratio = above ? nw.err / (nw.poll + nw.best) : NaN;
      numS.innerHTML = `<span>computed <b>${sci(ns.err)}</b></span><span class="capprox">best approximation ${sci(ns.best)}</span>`;
      numW.innerHTML = `<span>computed <b>${sci(nw.err)}</b></span><span class="capprox">best ${sci(nw.best)}</span>` +
        `<span class="cpoll">pollution ${sci(nw.poll)}</span>` +
        (isFinite(ratio) ? `<span>ratio C ≈ <b>${ratio.toFixed(2)}</b></span>`
                         : `<span>C: – (roundoff)</span>`);
    }
  }

  function draw(){
    PAL = null;
    layout();
    drawText();
    drawA(); drawB(); drawC();
    if(!raf) raf = requestAnimationFrame(loop);
  }
  function leave(){ if(raf) cancelAnimationFrame(raf); raf = null; }
  return {draw, leave};
}

/* ==================================================================== *
 *  4. Transient viewer                                                 *
 * ==================================================================== */
const TR_FIELDS = [['u', '|u<sub>h</sub>|'], ['sig', '|σ<sub>h</sub>|'], ['skw', '|skw σ<sub>h</sub>|']];
const TR_SCHEMES = [['afw', 'AFW₁', 'weak'], ['jm', 'JM₁', 'strong'], ['cg', 'SV₃ primal', 'approx']];
function Transient(host){
  const TD = window.TRANSIENT;
  host.classList.add('wd', 'tr-w');
  if(!TD){ host.textContent = 'transient data missing'; return {draw(){}}; }
  const nt = TD.times.length, nc = TD.mesh.nc;
  const mesh = {nc, xy: f32(TD.mesh.xy), tri: u32(TD.mesh.tri)};
  const Q = {};
  TR_SCHEMES.forEach(([s]) => { Q[s] = {}; TR_FIELDS.forEach(([f]) => { Q[s][f] = b64bytes(TD.schemes[s].fields[f].q); }); });
  const stepsDef = host.dataset.steps ? JSON.parse(host.dataset.steps) : [];
  const nearest = t => { let b = 0; TD.times.forEach((x, i) => { if(Math.abs(x - t) < Math.abs(TD.times[b] - t)) b = i; }); return b; };
  const st = {f: 'u', i: nearest(0.5), shared: false, playing: false, lastStep: -1};
  const m = /[?&]t=([\d.]+)/.exec(location.search);
  if(m) st.i = nearest(+m[1]);

  host.innerHTML = '';
  const bar = h('div', 'ctl'); bar.style.cssText = 'gap:14px;margin-bottom:8px';
  host.appendChild(bar);
  const play = h('button', 'play', '▶'); bar.appendChild(play);
  play.addEventListener('click', e => { e.stopPropagation(); toggle(); });
  const rng = h('input'); rng.type = 'range'; rng.min = 0; rng.max = nt - 1; rng.step = 1; rng.style.width = '340px';
  rng.addEventListener('input', () => { st.i = +rng.value; draw(); });
  bar.appendChild(rng);
  const tv = h('span', 'tval'); bar.appendChild(tv);
  const phase = h('span', 'tiny'); phase.style.minWidth = '210px'; bar.appendChild(phase);
  bar.appendChild(h('span', 'k', 'field'));
  const fSeg = seg(TR_FIELDS, k => { st.f = k; draw(); }); bar.appendChild(fSeg);
  bar.appendChild(h('span', 'k', 'scale'));
  const sSeg = seg([['1', 'shared'], ['0', 'per panel']], k => { st.shared = k === '1'; draw(); }); bar.appendChild(sSeg);

  const panes = h('div', 'panes'); host.appendChild(panes);
  const P3 = TR_SCHEMES.map(([s, lab, sym]) => {
    const p = h('div', 'pane');
    const t = h('div', 'ttl', `<span style="color:var(--${sym})">${lab}</span><span class="mx"></span>`);
    const cv = h('canvas');
    p.appendChild(t); p.appendChild(cv); panes.appendChild(p);
    return {s, t, cv, mx: t.querySelector('.mx')};
  });
  const cbar = h('canvas'); cbar.style.marginTop = '6px'; host.appendChild(cbar);
  const spark = h('canvas'); spark.style.marginTop = '4px'; host.appendChild(spark);

  let timer = null;
  // the frames are every step through the switch at t = 1 and every 5th
  // elsewhere; the switch (t = 1.00 to 1.15) also stays three times longer on
  // screen, so it plays in slow motion twice over
  const pace = i => TD.times[i] > 0.995 && TD.times[i] < 1.155 ? 600 : 200;
  function toggle(){
    st.playing = !st.playing;
    play.textContent = st.playing ? '❚❚' : '▶';
    if(st.playing){
      if(st.i >= nt - 1) st.i = 0;
      const tick = () => {
        st.i++;
        draw();
        if(st.i >= nt - 1) toggle();
        else timer = setTimeout(tick, pace(st.i));
      };
      timer = setTimeout(tick, pace(st.i));
    } else { clearTimeout(timer); timer = null; }
  }
  function value(s, f, i){
    const F = TD.schemes[s].fields[f], lo = F.lo[i], hi = F.hi[i], q = Q[s][f];
    return k => lo + q[i*nc + k] / 255 * (hi - lo);
  }
  function draw(){
    PAL = null;
    const P = pal(), W = host.clientWidth || 1212;
    const pw = Math.floor((W - 24) / 3);
    const side = Math.min(pw, (host.clientHeight || 600) - bar.offsetHeight - 26 - 32 - 100 - 24);
    rng.value = st.i; fSeg.set(st.f); sSeg.set(st.shared ? '1' : '0');
    const t = TD.times[st.i];
    tv.textContent = 't = ' + t.toFixed(2);
    phase.innerHTML = t <= 1 ? '<span class="dim">the lid is still accelerating, σ ≠ 0</span>'
                             : '<span class="cstrong">steady, and the exact σ is zero</span>';
    const his = TR_SCHEMES.map(([s]) => TD.schemes[s].fields[st.f].hi[st.i]);
    const shi = Math.max(...his);
    P3.forEach((p, j) => {
      const c = sizeCanvas(p.cv, side, side);
      // per panel, from the field's own min to its max: a stress that has relaxed
      // to 1e-7 still shows its pattern, instead of a flat square next to AFW's 65
      const lo = st.shared ? 0 : TD.schemes[p.s].fields[st.f].lo[st.i];
      const hi = st.shared ? shi : his[j], get = value(p.s, st.f, st.i);
      const vals = new Float32Array(nc);
      for(let k = 0; k < nc; k++) vals[k] = get(k);
      drawField(c, mesh, vals, 0, 0, side, v => cmap(hi > lo ? (v - lo) / (hi - lo) : 0));
      drawEdges(c, mesh, 0, 0, side, P.ink, 0.13);
      p.mx.textContent = 'max ' + sci(his[j]);
    });
    // horizontal colourbar
    const cw = 3*side + 24;
    const c2 = sizeCanvas(cbar, cw, 26);
    const gx = 60, gw = cw - 120;
    for(let x = 0; x < gw; x++){ c2.fillStyle = cmap(x / (gw - 1)); c2.fillRect(gx + x, 3, 1.2, 10); }
    c2.strokeStyle = P.rule2; c2.strokeRect(gx, 3, gw, 10);
    text(c2, gx - 6, 8, st.shared ? '0' : 'min', P.muted, 11, 'right');
    text(c2, gx + gw + 6, 8, st.shared ? sci(shi) : 'max', P.muted, 11, 'left');
    text(c2, gx + gw/2, 21, st.shared ? 'one colour scale for all three schemes' : 'each panel on its own scale, from its min to its max', P.faint, 10.5);
    drawSpark(cw);
  }
  function drawSpark(cw){
    const P = pal(), hh = 96;
    const c = sizeCanvas(spark, cw, hh);
    const L = 60, R = 60, T = 22, B = 18, w = cw - L - R, ph = hh - T - B;
    const e0 = -25, e1 = 2;
    const X = t => L + t / 2 * w, Y = v => T + (e1 - Math.log10(Math.max(v, 1e-25))) / (e1 - e0) * ph;
    c.fillStyle = 'rgba(147,161,161,0.10)'; c.fillRect(X(0), T, X(1) - X(0), ph);
    text(c, X(0.5), T + ph - 7, 'lid accelerating', P.faint, 10.5);
    c.strokeStyle = P.rule2; c.lineWidth = 1; c.strokeRect(L, T, w, ph);
    [-24, -16, -8, 0].forEach(e => { text(c, L - 5, Y(Math.pow(10, e)), dec(e), P.muted, 10, 'right'); });
    [0, 0.5, 1, 1.5, 2].forEach(t => text(c, X(t), hh - 7, 't = ' + t, P.muted, 10));
    text(c, L, T - 11, '‖skw σₕ‖ in L², every time step', P.ink2, 11, 'left');
    const S_ = TD.series;
    TR_SCHEMES.forEach(([s, lab, sym]) => {
      c.strokeStyle = P[sym]; c.lineWidth = 2;
      c.beginPath();
      S_[s].skw.forEach((v, i) => { const x = X(S_.t[i]), y = Y(v); i ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.stroke();
      const last = S_[s].skw[S_[s].skw.length - 1];
      text(c, L + w + 5, Y(last), lab, P[sym], 10.5, 'left');
    });
    const xc = X(TD.times[st.i]);
    c.strokeStyle = P.ink2; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(xc, T); c.lineTo(xc, T + ph); c.stroke(); c.setLineDash([]);
  }
  function setStep(k){
    if(k === st.lastStep) return;
    st.lastStep = k;
    const d = stepsDef[Math.min(k, stepsDef.length - 1)];
    if(d && !/[?&]t=/.test(location.search)){ st.f = d.f; st.i = nearest(d.t); }
  }
  function leave(){ if(st.playing) toggle(); st.lastStep = -1; }
  return {draw, setStep, leave, key: w => { if(w === 'play') toggle(); }};
}

/* ==================================================================== *
 *  5. Rods in a loaded network: non-separated vs separated stress      *
 * ==================================================================== */
const ROD_FORMS = [['nonsep', 'non-separated', 'AFW₁', 'approx'],
                   ['weak', 'separated, weak', 'AFW₁ + CG₁', 'weak'],
                   ['strong', 'separated, strong', 'JM₁ + CG₁', 'strong']];
/* how far a rod is turned from another: rods have no head, so modulo pi */
function rodOff(a, b){
  const d = ((a - b) % Math.PI + Math.PI) % Math.PI;
  return Math.min(d, Math.PI - d);
}
function Rods(host){
  const RD = (window.TORQUE || {}).rods;
  host.classList.add('wd', 'tr-w');
  if(!RD){ host.textContent = 'rod data missing: run make torque'; return {draw(){}}; }
  const n = RD.n, np = n*n, nt = RD.times.length;
  const exact = f32(RD.exact), TH = {}, MEAN = {};
  ROD_FORMS.forEach(([f]) => {
    TH[f] = RD.theta[f].map(f32);
    // mean misalignment from the exact rods, per load and frame
    MEAN[f] = TH[f].map(th => RD.times.map((_, i) => {
      let s = 0;
      for(let q = 0; q < np; q++) s += rodOff(th[i*np + q], exact[i*np + q]);
      return s / np;
    }));
  });
  const stepsDef = host.dataset.steps ? JSON.parse(host.dataset.steps) : [];
  const st = {li: 0, i: 0, playing: false, lastStep: -1};

  host.innerHTML = '';
  const bar = h('div', 'ctl'); bar.style.cssText = 'gap:14px;margin-bottom:8px';
  host.appendChild(bar);
  const play = h('button', 'play', '▶'); bar.appendChild(play);
  play.addEventListener('click', e => { e.stopPropagation(); toggle(); });
  const rng = h('input'); rng.type = 'range'; rng.min = 0; rng.max = nt - 1; rng.step = 1; rng.style.width = '300px';
  rng.addEventListener('input', () => { st.i = +rng.value; draw(); });
  bar.appendChild(rng);
  const tv = h('span', 'tval'); bar.appendChild(tv);
  bar.appendChild(h('span', 'k', 'symmetric load'));
  const lSeg = seg(RD.loads.map((l, j) => [String(j), 'L = ' + l]), k => { st.li = +k; draw(); });
  bar.appendChild(lSeg);

  const panes = h('div', 'panes'); host.appendChild(panes);
  const P3 = ROD_FORMS.map(([f, lab, sp, col]) => {
    const p = h('div', 'pane');
    const t = h('div', 'ttl', `<span><span style="color:var(--${col})">${lab}</span> <span class="dim">${sp}</span></span><span class="mx"></span>`);
    const cv = h('canvas');
    p.appendChild(t); p.appendChild(cv); panes.appendChild(p);
    return {f, cv, mx: t.querySelector('.mx')};
  });
  const note = h('div', 'tiny dim', 'The field turns the rods with the torque ρτ = −χ sin 2(θ − θ<sub>E</sub>), which they pass on to the ' +
    'network, so γ₁∂ₜθ = ρτ = 2s. Thin lines are the exact rods, the same for every L because a symmetric stress exerts no torque. ' +
    'Thick lines are the computed rods, γ₁∂ₜθₕ = 2sₕ, redder the further they drift. CG₂ director, level 2.');
  note.style.marginTop = '6px'; host.appendChild(note);
  const spark = h('canvas'); spark.style.marginTop = '4px'; host.appendChild(spark);

  let timer = null;
  function toggle(){
    st.playing = !st.playing;
    play.textContent = st.playing ? '❚❚' : '▶';
    if(st.playing){
      if(st.i >= nt - 1) st.i = 0;
      timer = setInterval(() => { st.i++; if(st.i >= nt - 1){ st.i = nt - 1; toggle(); } draw(); }, 150);
    } else { clearInterval(timer); timer = null; }
  }
  const rod = (c, x, y, a, len) => {
    const dx = Math.cos(a) * len / 2, dy = Math.sin(a) * len / 2;
    c.beginPath(); c.moveTo(x - dx, y + dy); c.lineTo(x + dx, y - dy); c.stroke();
  };
  function draw(){
    PAL = null;
    const P = pal(), W = host.clientWidth || 1212;
    const pw = Math.floor((W - 24) / 3);
    const side = Math.min(pw, (host.clientHeight || 600) - bar.offsetHeight - 26 - 40 - 96 - 20);
    rng.value = st.i; lSeg.set(String(st.li));
    tv.textContent = 't = ' + RD.times[st.i].toFixed(2);
    const gap = side / n, len = gap * 0.78, base = st.i * np;
    P3.forEach(p => {
      const c = sizeCanvas(p.cv, side, side), th = TH[p.f][st.li];
      c.fillStyle = P.paper; c.fillRect(0, 0, side, side);
      c.lineCap = 'round';
      for(let q = 0; q < np; q++){
        const x = (q % n + 0.5) * gap, y = side - (Math.floor(q / n) + 0.5) * gap;
        c.strokeStyle = P.rule2; c.lineWidth = 1.3;
        rod(c, x, y, exact[base + q], len);
        c.strokeStyle = mixc(P.ink2, P.red, Math.min(1, rodOff(th[base + q], exact[base + q]) / (Math.PI / 6)));
        c.lineWidth = 2.6;
        rod(c, x, y, th[base + q], len);
      }
      p.mx.textContent = (MEAN[p.f][st.li][st.i] * 180 / Math.PI).toFixed(1) + '° off';
    });
    drawSpark(3*side + 24);
  }
  function drawSpark(cw){
    const P = pal(), hh = 96;
    const c = sizeCanvas(spark, cw, hh);
    const L = 60, R = 130, T = 22, B = 18, w = cw - L - R, ph = hh - T - B;
    const tmax = RD.times[nt - 1];
    const X = t => L + t / tmax * w, Y = v => T + (1 - Math.min(v, Math.PI/4) / (Math.PI/4)) * ph;
    c.strokeStyle = P.rule2; c.lineWidth = 1; c.strokeRect(L, T, w, ph);
    [0, 15, 30, 45].forEach(d => text(c, L - 5, Y(d * Math.PI / 180), d + '°', P.muted, 10, 'right'));
    for(let t = 0; t <= tmax + 1e-9; t += 0.5) text(c, X(t), hh - 7, 't = ' + t, P.muted, 10);
    text(c, L, T - 11, 'mean angle between computed and exact rods, L = ' + RD.loads[st.li], P.ink2, 11, 'left');
    ROD_FORMS.forEach(([f, lab, , col], j) => {
      const s = MEAN[f][st.li];
      c.strokeStyle = P[col]; c.lineWidth = 2;
      c.beginPath();
      s.forEach((v, i) => { const x = X(RD.times[i]), y = Y(v); i ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.stroke();
      // a fixed legend: at L = 0 the three curves coincide
      c.fillStyle = P[col]; c.fillRect(L + w + 8, T + 6 + j*15 - 1, 14, 2.5);
      text(c, L + w + 27, T + 6 + j*15, lab, P[col], 10.5, 'left');
    });
    const xc = X(RD.times[st.i]);
    c.strokeStyle = P.ink2; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(xc, T); c.lineTo(xc, T + ph); c.stroke(); c.setLineDash([]);
  }
  function setStep(k){
    if(k === st.lastStep) return;
    st.lastStep = k;
    const li = stepsDef[Math.min(k, stepsDef.length - 1)];
    if(li != null){ st.li = li; st.i = 0; }
  }
  function leave(){ if(st.playing) toggle(); st.lastStep = -1; }
  return {draw, setStep, leave, key: w => { if(w === 'play') toggle(); }};
}

/* ==================================================================== *
 *  registry                                                            *
 * ==================================================================== */
const REG = {plot: Plot, matrix: Matrix, lab: Lab, transient: Transient, rods: Rods};
const live = new Map();
function inst(host){
  let i = live.get(host);
  if(!i){
    const f = REG[host.dataset.widget];
    if(!f) return null;
    try { i = f(host); } catch(err){ console.error(err); host.textContent = 'widget error: ' + err.message; return null; }
    live.set(host, i);
  }
  return i;
}
function show(host, step){
  const i = inst(host);
  if(!i) return;
  if(i.setStep) i.setStep(step);
  try { i.draw(); } catch(err){ console.error(err); }
}
function leave(slide){
  slide.querySelectorAll('[data-widget]').forEach(hh => { const i = live.get(hh); if(i && i.leave) i.leave(); });
}
function resize(slide){
  slide.querySelectorAll('[data-widget]').forEach(hh => { const i = live.get(hh); if(i) i.draw(); });
}
function retheme(){ PAL = null; live.forEach(i => i.draw()); }
function key(slide, what){
  slide.querySelectorAll('[data-widget]').forEach(hh => { const i = live.get(hh); if(i && i.key) i.key(what); });
}
return {show, leave, resize, retheme, key};
})();
