/**
 * Visual Traffic System — App Controller v3
 * BULLETPROOF: display:none/block for section visibility
 * All init steps wrapped in try-catch
 */
const API = '/api';

const PIPELINE_STAGES = [
  { id:'detect',name:'Detection\n& Tracking',icon:'\u{1F3AF}',desc:'YOLO11m + ByteTrack. 6 vehicle classes with per-class confidence thresholds.' },
  { id:'bev',name:'BEV\nTransform',icon:'\u{1F5FA}',desc:'Homography perspective warp. 4-point calibration maps tilted camera to bird\'s-eye world coords.' },
  { id:'trajectory',name:'Trajectory\nManagement',icon:'\u{1F4C8}',desc:'Deque-based trail history with EMA smoothing of positions and headings. Stride-adaptive alpha.' },
  { id:'motion',name:'Motion\nAnalysis',icon:'\u{1F3C3}',desc:'Speed over fixed windows. Classifies moving/stationary/parked with edge-margin heuristics.' },
  { id:'phi',name:'Phi\nIndex',icon:'\u{1F4CA}',desc:'Phi = w_rho*min(1,N/Nsat) + w_v*max(0,1-v/vref). Triggers conflict analysis above threshold.' },
  { id:'conflict',name:'Directional Field\nConvolution',icon:'\u{1F30A}',desc:'O(G^2) patented algorithm. 12 direction bins x 24 conflict pairs. GPU via cv2.filter2D.' },
  { id:'attribution',name:'Vehicle\nAttribution',icon:'\u{1F3AF}',desc:'Influence_i = R_k(P_i) * sum(R_k\'(P_i)). O(1) incremental ablation per vehicle.' },
  { id:'root_cause',name:'Root Cause\nTracing',icon:'\u{1F4A7}',desc:'Water Drop Propagation: sparse matrix, row-normalize, iter x+=alpha*A^T*x. Top-2 = root cause.' },
  { id:'visualization',name:'Visualization\nOutput',icon:'\u{1F3A8}',desc:'1920x1080 3-row composite: video+data | BEV+vehicles | Phi timeline.' },
];

const ALGO_STEPS = [
  { num:'1',title:'Scatter to Grid',brief:'Discretize vehicles into GxG occupancy, velocity, and direction fields.',detail:'Vehicles mapped to grid cells by world coords. Same-cell vehicles: speeds averaged, headings vector-averaged. Output: O(x,y), V(x,y), Theta(x,y).' },
  { num:'2',title:'Direction Binning',brief:'12 directional bins (30 deg each) with soft Gaussian weight assignment.',detail:'Continuous heading -> soft assignment to adjacent bins. sigma=bin_size/3 smooths transitions. Output: 12 occupancy layers O_0..O_11.' },
  { num:'3',title:'Anisotropic Kernels',brief:'Direction-stretched Gaussian: 7m forward, 1.4m lateral (3-sigma).',detail:'sigma_along=3.0, sigma_perp=0.6. Fan-shaped widening at distance. Backward decay 3x faster. 12 kernels pre-built once.' },
  { num:'4',title:'Influence Convolution',brief:'R_k = O_k (*) K_k via cv2.filter2D. GPU-accelerated.',detail:'Each direction bin convolved with its anisotropic kernel. Result: influence field showing "zone of influence" for vehicles heading direction k.' },
  { num:'5',title:'Conflict Field',brief:'C(x,y) = sum(R_a * R_b) across 24 conflict pairs.',detail:'6 opposite (180 deg) + 6 orthogonal (90 deg) + 12 same-direction (0 deg). Product peaks at influence field intersections.' },
  { num:'6',title:'Vehicle Attribution',brief:'Score each vehicle\'s congestion contribution from conflict field values at its position.',detail:'Influence_i = R_k(self at P_i) * sum(R_k\'(conflict at P_i)). Results ranked for ablation validation and root cause analysis.' },
];

const PRESETS = {
  quick:{label:'\u{1F680} Quick Start',imgsz:1280,conf:0.22,iou:0.40,ablation:true,quality:'balanced'},
  quality:{label:'\u{1F3AF} High Quality',imgsz:1600,conf:0.15,iou:0.35,ablation:true,quality:'quality'},
  fast:{label:'⚡ Fast Preview',imgsz:960,conf:0.30,iou:0.45,ablation:false,quality:'fast'},
};

const SECTION_ORDER = ['hero','pipeline','algorithm','launcher','monitor','architecture'];

const state = {
  currentSection:'hero',
  sites:[],selectedSite:null,selectedVideo:null,selectedModel:null,
  selectedPreset:'quick',running:false,
  phiHistory:[],monitorInterval:null,algoStep:0,algoAnimFrame:null,
};

// ═══════════════════════════════════════════════════════════════
// NAVIGATION — display:none/block (100% reliable)
// ═══════════════════════════════════════════════════════════════

function navigateTo(targetId) {
  if (targetId === state.currentSection) {
    console.log('[nav] already on:', targetId);
    return;
  }
  console.log('[nav]', state.currentSection, '->', targetId);

  try {
    // Find the two section DOM elements inside the viewport
    var vp = document.getElementById('main-viewport');
    if (!vp) { console.error('[nav] viewport not found'); return; }

    var oldSec = vp.querySelector('[data-section="' + state.currentSection + '"]');
    var newSec = vp.querySelector('[data-section="' + targetId + '"]');

    if (!oldSec) { console.error('[nav] old section not found:', state.currentSection); return; }
    if (!newSec) { console.error('[nav] new section not found:', targetId); return; }

    console.log('[nav] old:', oldSec.id, 'new:', newSec.id);

    // ── Update sidebar buttons ──
    var items = document.querySelectorAll('.sidebar-nav .nav-item');
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      if (item.dataset.section === targetId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    }

    // ── Switch sections: display:none old, display:block new ──
    // Step 1: Hide old section immediately
    oldSec.classList.remove('active');

    // Step 2: Show new section (starts with opacity:0 from .view-section default)
    newSec.classList.add('active');
    newSec.scrollTop = 0;

    // Step 3: Fade in new section (the CSS transition handles this)
    // The .view-section.active has opacity:1 + transition
    // We force a reflow then let the transition happen
    newSec.offsetHeight;

    state.currentSection = targetId;
    console.log('[nav] switched to:', targetId);

    // Trigger reveal animations
    triggerReveals(newSec);

    // Init chart if monitor section
    if (targetId === 'monitor') {
      setTimeout(function() {
        var c = document.getElementById('phi-chart');
        if (c && c.parentElement) {
          c.width = c.parentElement.clientWidth;
          c.height = c.parentElement.clientHeight;
          drawPhiChart();
        }
      }, 150);
    }
  } catch(e) {
    console.error('[nav] ERROR:', e.message, e.stack);
  }
}

// ═══════════════════════════════════════════════════════════════
// Reveal animations
// ═══════════════════════════════════════════════════════════════

function triggerReveals(section) {
  try {
    var reveals = section.querySelectorAll('.reveal');
    for (var i = 0; i < reveals.length; i++) {
      reveals[i].classList.remove('visible');
    }
    // Force reflow
    section.offsetHeight;
    // Add visible with stagger
    setTimeout(function() {
      for (var i = 0; i < reveals.length; i++) {
        var el = reveals[i];
        var delay = 0;
        if (el.classList.contains('delay-1')) delay = 80;
        else if (el.classList.contains('delay-2')) delay = 160;
        else if (el.classList.contains('delay-3')) delay = 240;
        else if (el.classList.contains('delay-4')) delay = 320;
        else if (el.classList.contains('delay-5')) delay = 400;
        else if (el.classList.contains('delay-6')) delay = 480;
        (function(el, d) {
          setTimeout(function() { el.classList.add('visible'); }, d);
        })(el, delay);
      }
    }, 50);
  } catch(e) { console.error('[reveal] error:', e.message); }
}

// ═══════════════════════════════════════════════════════════════
// Sidebar click handlers
// ═══════════════════════════════════════════════════════════════

function initSidebar() {
  try {
    var items = document.querySelectorAll('.sidebar-nav .nav-item');
    console.log('[init] sidebar items found:', items.length);
    for (var i = 0; i < items.length; i++) {
      (function(item) {
        item.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          var sec = this.dataset.section;
          console.log('[sidebar] clicked:', sec);
          if (sec) navigateTo(sec);
        });
      })(items[i]);
    }

    // Keyboard: 1-6
    document.addEventListener('keydown', function(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      var num = parseInt(e.key);
      if (num >= 1 && num <= SECTION_ORDER.length) {
        navigateTo(SECTION_ORDER[num - 1]);
      }
    });
  } catch(e) { console.error('[init] sidebar error:', e.message); }
}

// ═══════════════════════════════════════════════════════════════
// Hero buttons
// ═══════════════════════════════════════════════════════════════

function initHeroButtons() {
  try {
    var wrap = document.getElementById('hero-actions');
    if (!wrap) { console.error('[init] hero-actions not found'); return; }
    var btns = wrap.querySelectorAll('button[data-nav]');
    console.log('[init] hero buttons found:', btns.length);
    for (var i = 0; i < btns.length; i++) {
      (function(btn) {
        btn.addEventListener('click', function() {
          var target = this.dataset.nav;
          console.log('[hero] clicked nav to:', target);
          if (target) navigateTo(target);
        });
      })(btns[i]);
    }
  } catch(e) { console.error('[init] hero buttons error:', e.message); }
}

function initLaunchButtons() {
  try {
    var launchBtn = document.getElementById('launch-btn');
    var stopBtn = document.getElementById('stop-btn');
    if (launchBtn) {
      launchBtn.addEventListener('click', function() {
        console.log('[launch] Launch Pipeline clicked');
        launchPipeline();
      });
      console.log('[init] launch-btn handler attached');
    } else {
      console.error('[init] launch-btn not found');
    }
    if (stopBtn) {
      stopBtn.addEventListener('click', function() {
        console.log('[launch] Stop Pipeline clicked');
        stopPipeline();
      });
      console.log('[init] stop-btn handler attached');
    }
  } catch(e) { console.error('[init] launch buttons error:', e.message); }
}

// ═══════════════════════════════════════════════════════════════
// Pipeline Flow
// ═══════════════════════════════════════════════════════════════

function buildPipelineFlow() {
  try {
    var c = document.getElementById('pipeline-flow');
    if (!c) return;
    c.innerHTML = '';
    PIPELINE_STAGES.forEach(function(stage, i) {
      var step = document.createElement('div');
      step.className = 'pipeline-step';
      step.innerHTML = '<span class="step-icon">' + stage.icon + '</span><span class="step-name">' + stage.name + '</span>';
      step.addEventListener('click', function() { showPipelineDetail(i, step); });
      c.appendChild(step);
      if (i < PIPELINE_STAGES.length - 1) {
        var arr = document.createElement('span');
        arr.className = 'pipeline-arrow';
        arr.textContent = '→';
        c.appendChild(arr);
      }
    });
  } catch(e) { console.error('[init] pipeline error:', e.message); }
}

function showPipelineDetail(index, element) {
  try {
    var steps = document.querySelectorAll('.pipeline-step');
    for (var i = 0; i < steps.length; i++) steps[i].classList.remove('active');
    element.classList.add('active');
    var stage = PIPELINE_STAGES[index];
    var panel = document.getElementById('pipeline-detail');
    if (!panel) return;
    document.getElementById('pipeline-detail-icon').textContent = stage.icon;
    document.getElementById('pipeline-detail-title').textContent = stage.name.replace(/\n/g, ' — ');
    document.getElementById('pipeline-detail-body').textContent = stage.desc;
    panel.style.display = 'block';
  } catch(e) { console.error('[pipeline] detail error:', e.message); }
}

// ═══════════════════════════════════════════════════════════════
// Algorithm Steps
// ═══════════════════════════════════════════════════════════════

function buildAlgoSteps() {
  try {
    var c = document.getElementById('algo-steps');
    if (!c) return;
    c.innerHTML = '';
    ALGO_STEPS.forEach(function(step, i) {
      var div = document.createElement('div');
      div.className = 'algo-step';
      div.innerHTML = '<div class="step-num">' + step.num + '</div><div class="step-body"><div class="step-title">' + step.title + '</div><div class="step-brief">' + step.brief + '</div><div class="step-detail">' + step.detail + '</div></div>';
      div.addEventListener('click', function() {
        var was = div.classList.contains('expanded');
        var all = c.querySelectorAll('.algo-step');
        for (var j = 0; j < all.length; j++) all[j].classList.remove('expanded');
        if (!was) { div.classList.add('expanded'); state.algoStep = i; }
        else { state.algoStep = -1; }
        drawAlgoVisualization(state.algoStep);
      });
      c.appendChild(div);
    });
    var first = c.querySelector('.algo-step');
    if (first) first.classList.add('expanded');
    state.algoStep = 0;
    setTimeout(function() { drawAlgoVisualization(0); }, 200);
  } catch(e) { console.error('[init] algo steps error:', e.message); }
}

function drawAlgoVisualization(stepIdx) {
  var canvas = document.getElementById('algo-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var S = canvas.width;
  if (state.algoAnimFrame) cancelAnimationFrame(state.algoAnimFrame);
  ctx.clearRect(0, 0, S, S);
  ctx.fillStyle = '#141416'; ctx.fillRect(0, 0, S, S);

  // Grid lines
  var gs = 8, cs = S / gs;
  ctx.strokeStyle = 'rgba(51,51,56,0.3)'; ctx.lineWidth = 0.5;
  for (var i = 0; i <= gs; i++) {
    ctx.beginPath(); ctx.moveTo(i*cs, 0); ctx.lineTo(i*cs, S); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i*cs); ctx.lineTo(S, i*cs); ctx.stroke();
  }
  if (stepIdx < 0) {
    ctx.fillStyle = '#5c5c62'; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Click a step to visualize', S/2, S/2); return;
  }

  var veh = [{x:0.25,y:0.30,h:0.15},{x:0.55,y:0.25,h:0.85},{x:0.40,y:0.55,h:0.5},{x:0.70,y:0.60,h:0.2},{x:0.30,y:0.70,h:0.9},{x:0.60,y:0.45,h:0.65},{x:0.45,y:0.80,h:0.3}];

  if (stepIdx === 0) {
    veh.forEach(function(v) { ctx.beginPath(); ctx.arc(v.x*S,v.y*S,10,0,Math.PI*2); ctx.fillStyle='rgba(91,155,213,0.2)'; ctx.fill(); });
    veh.forEach(function(v) { ctx.beginPath(); ctx.arc(v.x*S,v.y*S,4,0,Math.PI*2); ctx.fillStyle='#5b9bd5'; ctx.fill(); });
    ctx.fillStyle='#e4e4e8';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('O(x,y) Occupancy Field',S/2,S-14);
  } else if (stepIdx === 1) {
    var cols=['#FF6B6B','#FF9F43','#FECA57','#54A0FF','#5F27CD','#01A3A4','#F368E0'];
    veh.forEach(function(v,i) {
      var cx=v.x*S,cy=v.y*S,a=v.h*Math.PI*2-Math.PI/2;
      ctx.save();ctx.translate(cx,cy);ctx.rotate(a);
      ctx.beginPath();ctx.moveTo(13,0);ctx.lineTo(-7,-5);ctx.lineTo(-7,5);ctx.closePath();
      ctx.fillStyle=cols[i%cols.length];ctx.globalAlpha=0.7;ctx.fill();ctx.restore();
    });
    ctx.fillStyle='#e4e4e8';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('12 Direction Bins',S/2,S-14);
  } else if (stepIdx === 2) {
    veh.slice(0,4).forEach(function(v,i) {
      var cx=v.x*S,cy=v.y*S,a=v.h*Math.PI*2;
      ctx.save();ctx.translate(cx,cy);ctx.rotate(a);ctx.beginPath();ctx.ellipse(18,0,28,7,0,0,Math.PI*2);
      ctx.strokeStyle=i<2?'rgba(91,155,213,0.6)':'rgba(212,149,107,0.6)';ctx.lineWidth=2;ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([]);ctx.restore();
      ctx.beginPath();ctx.arc(cx,cy,3.5,0,Math.PI*2);ctx.fillStyle='#5b9bd5';ctx.fill();
    });
    ctx.fillStyle='#e4e4e8';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Anisotropic Kernels',S/2,S-14);
  } else if (stepIdx === 3 || stepIdx === 4) {
    var imgData = ctx.createImageData(S, S);
    for (var y=0;y<S;y++){for(var x=0;x<S;x++){
      var idx=(y*S+x)*4,i2=0;
      veh.forEach(function(v,j){
        var dx=x/S-v.x,dy=y/S-v.y,along=dx*Math.cos(v.h*Math.PI*2)+dy*Math.sin(v.h*Math.PI*2),perp=-dx*Math.sin(v.h*Math.PI*2)+dy*Math.cos(v.h*Math.PI*2);
        var aDist=Math.sqrt((along/3)*(along/3)+(perp/0.6)*(perp/0.6));
        i2+=(stepIdx===4&&j>0)?Math.exp(-aDist*aDist*8)*0.4:Math.exp(-aDist*aDist*8);
      });
      i2=Math.min(1,i2*(stepIdx===4?1.5:1));
      var r,g,b,t;
      if(i2<0.25){t=i2/0.25;r=30+30*t;g=30+40*t;b=40+80*t;}
      else if(i2<0.5){t=(i2-0.25)/0.25;r=60+60*t;g=70+80*t;b=120-40*t;}
      else if(i2<0.75){t=(i2-0.5)/0.25;r=120+80*t;g=150+50*t;b=80-40*t;}
      else{t=(i2-0.75)/0.25;r=200+55*t;g=200-50*t;b=40-10*t;}
      imgData.data[idx]=r;imgData.data[idx+1]=g;imgData.data[idx+2]=b;imgData.data[idx+3]=Math.round(i2*200+30);
    }}
    ctx.putImageData(imgData,0,0);
    veh.forEach(function(v){ctx.beginPath();ctx.arc(v.x*S,v.y*S,4.5,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='rgba(0,0,0,0.5)';ctx.lineWidth=1;ctx.stroke();});
    ctx.fillStyle='#e4e4e8';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(stepIdx===3?'Influence Field':'Conflict Field',S/2,S-14);
  } else if (stepIdx === 5) {
    var infs=[0.85,0.62,0.38,0.91,0.45,0.72,0.28];
    veh.forEach(function(v,i){
      var cx=v.x*S,cy=v.y*S,inf=infs[i],r=7+inf*18,c2=inf>0.7?'#d55b5b':inf>0.5?'#d4956b':'#5b9bd5';
      ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
      var a2=inf>0.7?'rgba(213,91,91,0.22)':inf>0.5?'rgba(212,149,107,0.22)':'rgba(91,155,213,0.22)';
      ctx.fillStyle=a2;ctx.fill();ctx.strokeStyle=c2;ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle='#e4e4e8';ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(Math.round(inf*100)+'%',cx,cy+4);
    });
    ctx.fillStyle='#e4e4e8';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Vehicle Attribution',S/2,S-14);
  }
}

// ═══════════════════════════════════════════════════════════════
// Launcher — Site/Video/Model
// ═══════════════════════════════════════════════════════════════

function loadSites() {
  fetch(API + '/sites').then(function(r) { return r.json(); }).then(function(d) {
    state.sites = d.sites || [];
    renderSiteCards();
    if (d.default_site && !state.selectedSite) selectSite(d.default_site);
  }).catch(function(e) { console.error('loadSites:', e); });
}

function renderSiteCards() {
  var grid = document.getElementById('site-grid');
  if (!grid) return;
  grid.innerHTML = '';
  state.sites.forEach(function(site) {
    var card = document.createElement('div');
    card.className = 'site-card' + (state.selectedSite === site.key ? ' selected' : '');
    card.addEventListener('click', function() { selectSite(site.key); });
    card.innerHTML = '<div class="card-img-wrap">' +
      (site.calibration_exists
        ? '<img src="' + API + '/calibration-image/' + site.key + '" alt="' + site.display_name + '" loading="lazy" onerror="this.parentElement.innerHTML=\'<div class=card-img-placeholder>\u{1F6A6}</div>\'">'
        : '<div class="card-img-placeholder">\u{1F6A6}</div>') +
      '</div><div class="card-info"><div class="card-name">' + site.display_name + '</div><div class="card-meta">' + site.video_count + ' video(s) &middot; ' + site.key + '</div></div><div class="card-badge">' + site.video_count + ' \u{1F4F9}</div>';
    grid.appendChild(card);
  });
}

function selectSite(key) {
  state.selectedSite = key; state.selectedVideo = null;
  renderSiteCards(); renderVideoList(); updateLaunchButton();
}

function renderVideoList() {
  var list = document.getElementById('video-list');
  if (!list) return;
  var site = state.sites.find(function(s) { return s.key === state.selectedSite; });
  if (!site || !site.videos.length) {
    list.innerHTML = '<div style="color:var(--text-dim);padding:20px;text-align:center;">← Select an intersection first</div>';
    return;
  }
  list.innerHTML = '';
  site.videos.forEach(function(v) {
    var div = document.createElement('div');
    div.className = 'video-item' + (state.selectedVideo === v.path ? ' selected' : '');
    div.addEventListener('click', function() { selectVideo(v.path); });
    div.innerHTML = '<span>\u{1F3AC}</span><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;">' + v.name + '</span><span style="font-size:0.76rem;color:var(--text-dim);">' + v.size_mb + ' MB</span>';
    list.appendChild(div);
  });
}

function selectVideo(path) { state.selectedVideo = path; renderVideoList(); updateLaunchButton(); }

function loadModels() {
  fetch(API + '/models').then(function(r) { return r.json(); }).then(function(d) {
    var sel = document.getElementById('model-select');
    if (!sel) return;
    sel.innerHTML = d.models.map(function(m) { return '<option value="' + m.path + '">' + m.name + ' (' + m.size_mb + ' MB)</option>'; }).join('');
    if (d.models.length > 0) state.selectedModel = d.models[0].path;
  }).catch(function(e) { console.error('loadModels:', e); });
}

function buildPresetChips() {
  var c = document.getElementById('preset-chips');
  if (!c) return;
  c.innerHTML = '';
  Object.keys(PRESETS).forEach(function(key) {
    var p = PRESETS[key];
    var btn = document.createElement('button');
    btn.className = 'preset-chip' + (state.selectedPreset === key ? ' active' : '');
    btn.innerHTML = p.label + '<span style="font-size:0.7rem;opacity:0.7;display:block;">imgsz=' + p.imgsz + ' &middot; conf=' + p.conf + '</span>';
    btn.addEventListener('click', function() { selectPreset(key); });
    c.appendChild(btn);
  });
}

function selectPreset(key) { state.selectedPreset = key; buildPresetChips(); }

function updateLaunchButton() {
  var btn = document.getElementById('launch-btn');
  if (btn) btn.disabled = !(state.selectedSite && state.selectedVideo && !state.running);
}

// ── Launch / Stop ──────────────────────────────────────────────

function launchPipeline() {
  if (!state.selectedSite || !state.selectedVideo) { alert('Please select an intersection and a video first.'); return; }
  var body = {
    site_key: state.selectedSite, video_path: state.selectedVideo,
    model_path: (document.getElementById('model-select')||{}).value || '',
    preset: state.selectedPreset,
    stride: parseInt((document.getElementById('stride-select')||{}).value || '3'),
    show_windows: ((document.getElementById('show-windows')||{}).value || '1') === '1',
    max_frames: parseInt((document.getElementById('max-frames')||{}).value || '0') || 0,
  };
  fetch(API + '/launch', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { showLog('Error: ' + d.error, 'error'); return; }
      state.running = true; state.phiHistory = [];
      updateLaunchUI(); startMonitoring();
      showLog('[' + new Date().toLocaleTimeString() + '] Pipeline launched: ' + d.video);
    }).catch(function(e) { showLog('Connection error: ' + e.message, 'error'); });
}

function stopPipeline() {
  fetch(API + '/stop', { method:'POST' }).then(function() {
    state.running = false; updateLaunchUI(); stopMonitoring();
    showLog('[' + new Date().toLocaleTimeString() + '] Pipeline stopped.', 'success');
  }).catch(function(e) { showLog('Error: ' + e.message, 'error'); });
}

function updateLaunchUI() {
  var lb = document.getElementById('launch-btn'), sb = document.getElementById('stop-btn');
  if (lb) lb.style.display = state.running ? 'none' : 'inline-flex';
  if (sb) sb.style.display = state.running ? 'inline-flex' : 'none';
  var st = document.getElementById('launch-status');
  if (st) st.innerHTML = state.running ? '<span style="color:var(--green);">● Pipeline running — OpenCV windows open</span>' : '';
  updateLaunchButton();
}

function showLog(msg, type) {
  var c = document.getElementById('log-console');
  if (!c) return;
  c.style.display = 'block';
  var line = document.createElement('div');
  line.className = 'log-line' + (type ? ' ' + type : '');
  line.textContent = msg;
  c.appendChild(line);
  c.scrollTop = c.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════
// Live Monitoring
// ═══════════════════════════════════════════════════════════════

function startMonitoring() {
  updateMonitorStatus('running', 'Running');
  if (state.monitorInterval) clearInterval(state.monitorInterval);
  state.monitorInterval = setInterval(pollMetrics, 1500);
  pollMetrics();
}

function stopMonitoring() {
  if (state.monitorInterval) { clearInterval(state.monitorInterval); state.monitorInterval = null; }
  updateMonitorStatus('idle', 'No pipeline running');
}

function pollMetrics() {
  fetch(API + '/status').then(function(r) { return r.json(); }).then(function(d) {
    if (!d.running) { state.running = false; updateLaunchUI(); stopMonitoring(); return; }
    var m = d.metrics || {};
    if (Object.keys(m).length > 0) { updateMetricTiles(m); updatePhiChart(m); }
  }).catch(function() {});
}

function updateMonitorStatus(type, text) {
  var pill = document.querySelector('#monitor-status .status-pill');
  if (pill) pill.className = 'status-pill ' + type;
  var dot = document.querySelector('#monitor-status .status-dot-sm');
  if (dot) dot.className = 'status-dot-sm ' + type;
  var el = document.getElementById('monitor-status-text');
  if (el) el.textContent = text;
}

function updateMetricTiles(m) {
  var phi = parseFloat(m.phi_t) || 0;
  var pe = document.getElementById('metric-phi');
  if (pe) { pe.textContent = phi.toFixed(4); pe.style.color = phi<0.3?'#6baf6b':phi<0.55?'#e2b96f':phi<0.75?'#d4956b':'#d55b5b'; }
  var ve = document.getElementById('metric-vehicles'); if (ve) ve.textContent = m.vehicle_total != null ? m.vehicle_total : '--';
  var se = document.getElementById('metric-speed'); if (se) se.textContent = parseFloat(m.avg_speed_mps||0).toFixed(2);
  var pe2 = document.getElementById('metric-parked'); if (pe2) pe2.textContent = m.parked_count != null ? m.parked_count : '--';
  if (m.screenshot) {
    var prev = document.getElementById('live-preview');
    if (prev) prev.innerHTML = '<img src="data:image/jpeg;base64,' + m.screenshot + '" alt="Live" style="width:100%;display:block;">';
  }
}

function updatePhiChart(m) {
  if (m.phi_t === undefined && m.phi_t === null) return;
  var now = state.phiHistory.length > 0 ? state.phiHistory[state.phiHistory.length-1].t + 2 : 0;
  state.phiHistory.push({t:now, phi:parseFloat(m.phi_t)||0});
  if (state.phiHistory.length > 60) state.phiHistory = state.phiHistory.slice(-60);
  drawPhiChart();
}

function drawPhiChart() {
  var canvas = document.getElementById('phi-chart');
  if (!canvas || !canvas.parentElement) return;
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  var ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  var pad = {top:18,right:18,bottom:28,left:42}, pw = W-pad.left-pad.right, ph = H-pad.top-pad.bottom;
  ctx.clearRect(0,0,W,H);
  if (state.phiHistory.length < 2) {
    ctx.fillStyle='#5c5c62';ctx.font='13px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Waiting for data...',W/2,H/2);return;
  }
  var pts = state.phiHistory, tMin = pts[0].t, tMax = pts[pts.length-1].t, tR = Math.max(tMax-tMin,1);
  ctx.strokeStyle='rgba(51,51,56,0.4)';ctx.lineWidth=0.5;
  for (var i=0;i<=4;i++) {
    var yy = pad.top+ph*i/4;
    ctx.beginPath();ctx.moveTo(pad.left,yy);ctx.lineTo(W-pad.right,yy);ctx.stroke();
    ctx.fillStyle='#5c5c62';ctx.font='9px Inter,sans-serif';ctx.textAlign='right';ctx.fillText((1-i/4).toFixed(2),pad.left-5,yy+3);
  }
  var ty = pad.top+ph*(1-0.70);
  ctx.beginPath();ctx.setLineDash([5,3]);ctx.moveTo(pad.left,ty);ctx.lineTo(W-pad.right,ty);ctx.strokeStyle='rgba(213,91,91,0.45)';ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#d55b5b';ctx.font='8px Inter,sans-serif';ctx.textAlign='left';ctx.fillText('threshold 0.70',W-pad.right-72,ty-3);
  var grad=ctx.createLinearGradient(0,pad.top,0,pad.top+ph);
  grad.addColorStop(0,'#d55b5b');grad.addColorStop(0.25,'#d4956b');grad.addColorStop(0.45,'#e2b96f');grad.addColorStop(1,'#6baf6b');
  ctx.beginPath();
  pts.forEach(function(p,i){var x=pad.left+(p.t-tMin)/tR*pw,y=pad.top+ph*(1-Math.min(1,Math.max(0,p.phi)));i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.strokeStyle=grad;ctx.lineWidth=2;ctx.lineJoin='round';ctx.stroke();
  ctx.lineTo(pad.left+pw,pad.top+ph);ctx.lineTo(pad.left,pad.top+ph);ctx.closePath();
  var fg=ctx.createLinearGradient(0,pad.top,0,pad.top+ph);
  fg.addColorStop(0,'rgba(213,91,91,0.12)');fg.addColorStop(0.5,'rgba(226,185,111,0.04)');fg.addColorStop(1,'rgba(107,175,107,0.01)');
  ctx.fillStyle=fg;ctx.fill();
  pts.forEach(function(p){var x=pad.left+(p.t-tMin)/tR*pw,y=pad.top+ph*(1-Math.min(1,Math.max(0,p.phi)));ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fillStyle=p.phi>0.75?'#d55b5b':p.phi>0.55?'#d4956b':p.phi>0.3?'#e2b96f':'#6baf6b';ctx.fill();});
  ctx.fillStyle='#5c5c62';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(tMin.toFixed(0)+'s',pad.left,H-8);ctx.fillText(tMax.toFixed(0)+'s',W-pad.right,H-8);
}

// ═══════════════════════════════════════════════════════════════
// Events
// ═══════════════════════════════════════════════════════════════

function loadEvents() {
  fetch(API + '/events').then(function(r) { return r.json(); }).then(function(d) {
    var events = d.events || [], c = document.getElementById('events-list');
    if (!c) return;
    if (!events.length) { c.innerHTML = '<div style="color:var(--text-dim);text-align:center;padding:14px;">No events yet.</div>'; return; }
    c.innerHTML = events.slice(0,10).map(function(ev) {
      return '<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid var(--border);"><span>\u{1F4CC}</span><div style="flex:1;"><div style="font-weight:600;font-size:0.88rem;">'+ev.event+' &mdash; '+ev.site+'/'+ev.video+'</div><div style="font-size:0.76rem;color:var(--text-dim);">Peak Φ: '+ev.peak_phi.toFixed(3)+' &middot; '+ev.duration_s.toFixed(1)+'s</div></div><span style="font-size:0.78rem;color:'+(ev.peak_phi>0.75?'var(--red)':ev.peak_phi>0.55?'var(--orange)':'var(--yellow)')+';">Φ '+ev.peak_phi.toFixed(2)+'</span></div>';
    }).join('');
  }).catch(function(e) { console.error('loadEvents:', e); });
}

// ═══════════════════════════════════════════════════════════════
// Hero Phi Animation
// ═══════════════════════════════════════════════════════════════

function animateHeroPhi() {
  var circle = document.getElementById('hero-phi-circle'), valEl = document.getElementById('hero-phi-val');
  if (!circle || !valEl) return;
  var circ = 2*Math.PI*58, pat=[{v:0.12,d:2200},{v:0.28,d:1600},{v:0.45,d:2000},{v:0.68,d:1800},{v:0.85,d:1600},{v:0.52,d:2000},{v:0.22,d:2400},{v:0.08,d:2800}];
  var pi=0,sv=0,tv=pat[0].v,st=performance.now(),dur=pat[0].d;
  function gc(phi){if(phi<0.3)return'#6baf6b';if(phi<0.55)return'#e2b96f';if(phi<0.75)return'#d4956b';return'#d55b5b';}
  function anim(now){
    var el=now-st;
    if(el>=dur){sv=tv;pi=(pi+1)%pat.length;tv=pat[pi].v;dur=pat[pi].d;st=now;el=0;}
    var t=el/dur,ease=t<0.5?2*t*t:-1+(4-2*t)*t,cur=sv+(tv-sv)*ease,off=circ*(1-cur),col=gc(cur);
    circle.setAttribute('stroke-dashoffset',off);circle.setAttribute('stroke',col);valEl.textContent=cur.toFixed(2);valEl.style.color=col;
    requestAnimationFrame(anim);
  }
  requestAnimationFrame(anim);
}

// ═══════════════════════════════════════════════════════════════
// Initialize
// ═══════════════════════════════════════════════════════════════

function init() {
  console.log('[init] Visual Traffic System starting...');

  try { initSidebar(); console.log('[init] sidebar OK'); } catch(e) { console.error('[init] sidebar FAIL:', e.message); }
  try { initHeroButtons(); console.log('[init] hero buttons OK'); } catch(e) { console.error('[init] hero buttons FAIL:', e.message); }
  try { initLaunchButtons(); console.log('[init] launch buttons OK'); } catch(e) { console.error('[init] launch buttons FAIL:', e.message); }
  try { buildPipelineFlow(); console.log('[init] pipeline OK'); } catch(e) { console.error('[init] pipeline FAIL:', e.message); }
  try { buildAlgoSteps(); console.log('[init] algo OK'); } catch(e) { console.error('[init] algo FAIL:', e.message); }
  try { buildPresetChips(); console.log('[init] presets OK'); } catch(e) { console.error('[init] presets FAIL:', e.message); }
  try { animateHeroPhi(); console.log('[init] phi anim OK'); } catch(e) { console.error('[init] phi anim FAIL:', e.message); }

  // Trigger reveals on the initial (hero) section
  try {
    var initSec = document.querySelector('.view-section.active');
    if (initSec) triggerReveals(initSec);
  } catch(e) { console.error('[init] reveals FAIL:', e.message); }

  // Load data
  try { loadSites(); } catch(e) { console.error('loadSites:', e.message); }
  try { loadModels(); } catch(e) { console.error('loadModels:', e.message); }
  try { loadEvents(); } catch(e) { console.error('loadEvents:', e.message); }
  setInterval(loadEvents, 10000);

  window.addEventListener('resize', function() { if (state.phiHistory.length > 1) drawPhiChart(); });

  // Initial chart size
  setTimeout(function() {
    var c = document.getElementById('phi-chart');
    if (c && c.parentElement) { c.width = c.parentElement.clientWidth; c.height = c.parentElement.clientHeight; }
  }, 600);

  console.log('[init] DONE. Navigate with sidebar buttons, hero buttons, or keys 1-6.');
}

// Global exports
window.launchPipeline = launchPipeline;
window.stopPipeline = stopPipeline;
window.navigateTo = navigateTo;
window.selectVideo = selectVideo;
window.selectPreset = selectPreset;

// Start
if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', init); }
else { init(); }
