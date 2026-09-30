/* ==================================================================== *
 *  Deck engine (from the Periodic Table deck)                          *
 * ==================================================================== */
function el(tag, cls, html){
  const n = document.createElement(tag);
  if(cls) n.className = cls;
  if(html != null) n.innerHTML = html;
  return n;
}
const stage = document.getElementById('stage');
const tpl = document.getElementById('deck');
stage.innerHTML = '';
const slides = [];

Array.from(tpl.content.querySelectorAll('section.slide')).forEach((s,i)=>{
  const w = el('div','wrap'); w.dataset.i = i;
  w.appendChild(s);
  stage.appendChild(w);
  slides.push(s);
});

/* figures: [data-fig] -> data URI from figures.js */
document.querySelectorAll('[data-fig]').forEach(img=>{
  const k = img.dataset.fig;
  if(window.FIG && window.FIG[k]) img.src = window.FIG[k];
  else img.replaceWith(el('div','tiny dim','[missing figure: '+k+']'));
});

/* mathematics: KaTeX, with the macros of p-and-m.tex / main.tex */
const KATEX_MACROS = {
  '\\vec':'\\boldsymbol{#1}', '\\tensor':'\\underline{\\underline{#1}}',
  '\\div':'\\operatorname{div}', '\\curl':'\\operatorname{curl}', '\\tr':'\\operatorname{tr}',
  '\\sym':'\\operatorname{sym}', '\\skw':'\\operatorname{skw}', '\\symgrad':'\\varepsilon',
  '\\Th':'\\mathcal{T}_h', '\\norm':'\\lVert #1\\rVert', '\\O':'\\mathcal{O}',
};
window.KATEX_MACROS = KATEX_MACROS;
if(window.renderMathInElement){
  slides.forEach(s=>renderMathInElement(s, {
    delimiters:[{left:'\\[',right:'\\]',display:true},{left:'$',right:'$',display:false}],
    macros:KATEX_MACROS, trust:true, strict:false, throwOnError:false,
  }));
}

/* citations: <cite data-ref="key"> -> a link to the DOI / arXiv page */
const BIBUSED = new Set();
function bibLink(e){
  if(e.doi) return 'https://doi.org/' + e.doi;
  if(e.arx) return 'https://arxiv.org/abs/' + e.arx;
  return e.url || null;
}
function bibShort(e){ return e.a + (e.y ? ' ' + e.y : ''); }
document.querySelectorAll('cite[data-ref]').forEach(node=>{
  const k = node.dataset.ref, e = (window.BIB||{})[k];
  if(!e){ node.replaceWith(document.createTextNode('[?' + k + ']')); return; }
  BIBUSED.add(k);
  const href = bibLink(e);
  // a <cite class="brk"> may break across lines; citations otherwise stay whole
  const a = el(href ? 'a' : 'span', 'ref' + (node.className ? ' ' + node.className : ''));
  a.innerHTML = node.innerHTML.trim() || bibShort(e);
  a.title = e.a + ' — ' + e.t + (e.j ? ', ' + e.j : '') + (e.y ? ' (' + e.y + ')' : '');
  if(href){ a.href = href; a.target = '_blank'; a.rel = 'noopener'; }
  node.replaceWith(a);
});
(function buildBib(){
  const list = document.getElementById('biblist');
  const keys = [...BIBUSED].sort((x,y)=>{
    const a = window.BIB[x], b = window.BIB[y];
    return (a.a + a.y).localeCompare(b.a + b.y);
  });
  keys.forEach(k=>{
    const e = window.BIB[k], href = bibLink(e), li = el('li');
    const src = [e.j, e.v && ('vol. ' + e.v), e.p, e.y].filter(Boolean).join(', ');
    li.innerHTML = '<b>' + e.a + '</b>. ' +
      (href ? '<a class="ti" href="' + href + '" target="_blank" rel="noopener">' + e.t + '</a>'
            : '<span class="ti">' + e.t + '</span>') +
      (src ? ' <span class="src">' + src + '</span>' : '') +
      (e.doi ? ' <span class="nolink">doi:' + e.doi + '</span>'
             : e.arx ? ' <span class="nolink">arXiv:' + e.arx + '</span>'
             : e.n ? ' <span class="nolink">' + e.n + '</span>' : '');
    list.appendChild(li);
  });
})();

/* fragment discovery */
const steps = slides.map(s=>{
  const f = Array.from(s.querySelectorAll('[data-frag]'));
  const groups = new Map();
  f.forEach(n=>{
    const k = parseInt(n.dataset.frag,10) || 1;
    if(!groups.has(k)) groups.set(k,[]);
    groups.get(k).push(n);
  });
  return [...groups.keys()].sort((a,b)=>a-b).map(k=>groups.get(k));
});

let cur = 0, step = 0, buf = '';
const wraps = Array.from(document.querySelectorAll('.wrap'));
const PRINT = location.search.includes('print');

function fit(){
  const ov = document.body.classList.contains('overview');
  const s = ov ? Math.min(0.19, (window.innerWidth-70)/6/1280)
               : Math.min(window.innerWidth/1280, window.innerHeight/720);
  document.documentElement.style.setProperty('--s', ov ? Math.max(s,0.14) : s);
}

/* widgets: mounted the first time their slide is shown (all of them in the
   overview and in print), then told about every build step */
function widgetsOn(slide){
  slide.querySelectorAll('[data-widget]').forEach(h=>{
    if(window.WIDGETS) WIDGETS.show(h, slides.indexOf(slide) === cur ? step : 99);
  });
}

function render(){
  wraps.forEach((w,i)=>w.classList.toggle('current', i===cur));
  const s = slides[cur];
  s.querySelectorAll('[data-frag]').forEach(n=>n.classList.remove('on'));
  for(let k=0;k<step;k++) (steps[cur][k]||[]).forEach(n=>n.classList.add('on'));
  widgetsOn(s);
  const sect = s.dataset.sect || '';
  document.getElementById('head').textContent = sect;
  document.getElementById('foot').innerHTML = '<b>'+(cur+1)+'</b> / '+slides.length;
  document.querySelector('#bar>i').style.width = (100*(cur)/(slides.length-1))+'%';
  if(!document.body.classList.contains('overview'))
    history.replaceState(null,'', location.search + '#'+(cur+1)+(step?('.'+step):''));
}

function go(i, st){
  if(window.WIDGETS) WIDGETS.leave(slides[cur]);
  cur = Math.max(0, Math.min(slides.length-1, i));
  const n = steps[cur].length;
  step = st === 'end' ? n : Math.max(0, Math.min(n, st|0));
  render();
}
function next(){ if(step < steps[cur].length) { step++; render(); }
                 else if(cur < slides.length-1) go(cur+1, 0); }
function prev(){ if(step > 0) { step--; render(); }
                 else if(cur > 0) go(cur-1, 'end'); }

/* a slider or button keeps no focus after the mouse lets go of it, so the
   clicker's arrow keys go back to stepping the deck */
addEventListener('pointerup', e=>{
  const t = e.target.closest && e.target.closest('input[type=range], button');
  if(t) setTimeout(()=>t.blur(), 0);
});
addEventListener('resize', ()=>{ fit(); if(window.WIDGETS) WIDGETS.resize(slides[cur]); });
addEventListener('keydown', e=>{
  if(e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if(e.target.closest && e.target.closest('input, textarea, select')) return;
  if(document.body.classList.contains('help') && k !== '?'){ document.body.classList.remove('help'); return; }
  if(/^[0-9]$/.test(k)){ buf += k; return; }
  if(k === 'Enter' && buf){ go(parseInt(buf,10)-1, 0); buf=''; return; }
  buf = '';
  switch(k){
    case 'ArrowRight': case ' ': case 'ArrowDown': case 'PageDown': case 'n':
      e.preventDefault(); e.shiftKey ? go(cur+1,0) : next(); break;
    case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'p':
      e.preventDefault(); e.shiftKey ? go(cur-1,0) : prev(); break;
    case 'Home': e.preventDefault(); go(0,0); break;
    case 'End':  e.preventDefault(); go(slides.length-1,'end'); break;
    case 'a': case 'A': go(cur,'end'); break;
    case 'k': case 'K': if(window.WIDGETS) WIDGETS.key(slides[cur], 'play'); break;
    case 'o': case 'O':
      document.body.classList.toggle('overview'); fit();
      document.body.classList.remove('bib');
      if(document.body.classList.contains('overview')){
        slides.forEach(widgetsOn);
        wraps[cur].scrollIntoView({block:'center'});
      } else if(window.WIDGETS) WIDGETS.resize(slides[cur]);
      break;
    case 'b': case 'B': document.body.classList.toggle('bib'); break;
    case 'f': case 'F':
      if(document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.(); break;
    case 'd': case 'D': {
      const r = document.documentElement;
      const now = r.dataset.theme ||
        (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
      r.dataset.theme = now === 'dark' ? 'light' : 'dark';
      if(window.WIDGETS) WIDGETS.retheme();
      break; }
    case '?': document.body.classList.toggle('help'); break;
    case 'Escape': document.body.classList.remove('overview','help','bib'); fit(); break;
  }
});
stage.addEventListener('click', e=>{
  if(!document.body.classList.contains('overview')) return;
  const w = e.target.closest('.wrap'); if(!w) return;
  document.body.classList.remove('overview'); fit(); go(+w.dataset.i, 0);
});
addEventListener('hashchange', ()=>{
  const m = /^#(\d+)(?:\.(\d+))?$/.exec(location.hash);
  if(m) go(+m[1]-1, m[2]?+m[2]:0);
});

if(PRINT){ document.body.classList.add('print'); }
const th = /theme=(light|dark)/.exec(location.search);
document.documentElement.dataset.theme = th ? th[1]
  : (document.documentElement.dataset.theme || 'light');
if(/[?&]bib\b/.test(location.search)) document.body.classList.add('bib');
fit();
const m0 = /^#(\d+)(?:\.(\d+))?$/.exec(location.hash);
go(m0 ? +m0[1]-1 : 0, m0 && m0[2] ? +m0[2] : 0);
if(PRINT) slides.forEach(widgetsOn);
