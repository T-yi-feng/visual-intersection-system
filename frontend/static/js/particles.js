/**
 * Intersection Traffic Flow Particle System v3
 *
 * Models N/S/E/W directional traffic flowing through a central intersection.
 * Particles spawn at road edges, travel toward center, cross or turn,
 * and exit at the opposite/far edge. Mouse creates "conflict zone" —
 * particles near cursor slow down, change color, and cluster visually.
 *
 * Road layout:
 *        N (top→bottom + bottom→top)
 *        |
 *   W ---+--- E (left→right + right→left)
 *        |
 *        S
 */
(function(){
var canvas = document.getElementById('particle-canvas');
var ctx = canvas.getContext('2d');
var W, H, particles = [], mouse = {x:-9999,y:-9999,active:false}, frame = 0;

// ── Road configuration ────────────────────────────────────────
var ROADS = [
  // North→South (top edge → bottom edge)
  { dir:'NS', spawnAngle:-Math.PI/2, spawnEdge:'top', exitEdge:'bottom' },
  // South→North (bottom edge → top edge)
  { dir:'SN', spawnAngle:Math.PI/2, spawnEdge:'bottom', exitEdge:'top' },
  // East→West (right edge → left edge)
  { dir:'EW', spawnAngle:0, spawnEdge:'right', exitEdge:'left' },
  // West→East (left edge → right edge)
  { dir:'WE', spawnAngle:Math.PI, spawnEdge:'left', exitEdge:'right' },
];

var CFG = {
  particleCount: 160,
  baseSpeed: 0.8,
  maxSpeed: 2.5,
  particleRadius: 2.2,
  glowRadius: 6,
  trailLength: 8,
  mouseRadius: 160,
  mouseForce: 0.06,
  connectionDist: 100,
  roadWidth: 70,              // lane spread width in pixels
  turnProbability: 0.25,      // chance of turning at intersection
  centerRadius: 50,           // intersection center zone radius
  colors: {
    free:      [107, 175, 107],
    moderate:  [226, 185, 111],
    heavy:     [212, 149, 107],
    severe:    [213, 91, 91],
    trail:     [91, 155, 213],
  },
};

// ── Particle ──────────────────────────────────────────────────

function Particle() {
  this.reset(true);
}

Particle.prototype.reset = function(initial) {
  // Pick a random road direction
  var road = ROADS[Math.floor(Math.random() * ROADS.length)];
  this.road = road;
  this.turning = false;
  this.turnProgress = 0;

  if (initial) {
    // Random position anywhere for smooth initial state
    this.x = Math.random() * W;
    this.y = Math.random() * H;
    // Assign a direction anyway
    this.heading = road.spawnAngle + (Math.random() - 0.5) * 0.3;
    this.speed = CFG.baseSpeed + Math.random() * 0.8;
    this.inIntersection = false;
  } else {
    // Spawn at the correct edge
    var laneOffset = (Math.random() - 0.5) * CFG.roadWidth;
    switch (road.spawnEdge) {
      case 'top':
        this.x = W/2 + laneOffset;
        this.y = -10;
        break;
      case 'bottom':
        this.x = W/2 + laneOffset;
        this.y = H + 10;
        break;
      case 'left':
        this.x = -10;
        this.y = H/2 + laneOffset;
        break;
      case 'right':
        this.x = W + 10;
        this.y = H/2 + laneOffset;
        break;
    }
    this.heading = road.spawnAngle + (Math.random() - 0.5) * 0.25;
    this.speed = CFG.baseSpeed + Math.random() * 1.2;
    this.inIntersection = false;
  }

  this.radius = CFG.particleRadius + Math.random() * 1.2;
  this.alpha = 0.35 + Math.random() * 0.45;
  this.trail = [];
  this.color = CFG.colors.free.slice();
  this.targetColor = CFG.colors.free.slice();
  this.colorTransition = 0;
  this.baseSpeed = this.speed;
};

Particle.prototype.update = function() {
  // Store trail
  this.trail.push({x:this.x, y:this.y, a:this.alpha});
  if (this.trail.length > CFG.trailLength) this.trail.shift();

  // ── Steering toward target ──────────────────────────────────
  var cx = W/2, cy = H/2;
  var distToCenter = Math.sqrt((this.x - cx)*(this.x - cx) + (this.y - cy)*(this.y - cy));

  // Desired heading: toward center (entry) or away from center toward exit (after crossing)
  var desiredHeading;
  if (!this.inIntersection) {
    // Heading toward center
    desiredHeading = Math.atan2(cy - this.y, cx - this.x);
    // Once near center, mark as inside intersection
    if (distToCenter < CFG.centerRadius) {
      this.inIntersection = true;
      // Random chance to turn
      if (Math.random() < CFG.turnProbability) {
        this.turning = true;
        // Pick a perpendicular exit direction
        var turns = [-Math.PI/2, Math.PI/2];
        this.turnAngle = turns[Math.floor(Math.random() * turns.length)];
      }
    }
  } else {
    // Heading away from center toward the exit edge
    if (this.turning && this.turnProgress < 1) {
      // Smooth turn
      this.turnProgress += 0.03;
      var baseExit = this.road.spawnAngle + Math.PI; // straight through
      desiredHeading = baseExit + this.turnAngle * this.turnProgress;
    } else {
      // Straight through
      desiredHeading = this.road.spawnAngle + Math.PI;
    }
  }

  // Smooth heading adjustment
  var headingDiff = desiredHeading - this.heading;
  // Normalize to [-PI, PI]
  while (headingDiff > Math.PI) headingDiff -= 2*Math.PI;
  while (headingDiff < -Math.PI) headingDiff += 2*Math.PI;
  this.heading += headingDiff * 0.04;

  // ── Mouse interaction (conflict zone) ───────────────────────
  var mdx = this.x - mouse.x, mdy = this.y - mouse.y;
  var md = Math.sqrt(mdx*mdx + mdy*mdy);

  if (mouse.active && md < CFG.mouseRadius) {
    var force = 1 - md/CFG.mouseRadius;
    // Repel from mouse
    this.heading += (mdx/(md+0.1)) * CFG.mouseForce * force * 0.5;
    // Slow down
    this.speed = this.baseSpeed * (1 - force * 0.6);
    // Color shift based on conflict intensity
    if (force > 0.7) this.targetColor = CFG.colors.severe;
    else if (force > 0.4) this.targetColor = CFG.colors.heavy;
    else this.targetColor = CFG.colors.moderate;
    this.colorTransition = Math.min(1, this.colorTransition + 0.04);
  } else {
    this.speed += (this.baseSpeed - this.speed) * 0.04;
    this.targetColor = CFG.colors.free;
    this.colorTransition = Math.max(0, this.colorTransition - 0.02);
  }

  // Interpolate color
  if (this.colorTransition > 0.005) {
    for (var i = 0; i < 3; i++) {
      this.color[i] = Math.round(this.color[i] + (this.targetColor[i] - this.color[i]) * this.colorTransition);
    }
  }

  // ── Move ────────────────────────────────────────────────────
  this.x += Math.cos(this.heading) * this.speed;
  this.y += Math.sin(this.heading) * this.speed;

  // Fade trail
  for (var i = 0; i < this.trail.length; i++) {
    this.trail[i].a = this.alpha * (i / CFG.trailLength) * 0.5;
  }

  // ── Check if exited ─────────────────────────────────────────
  var margin = 60;
  var exited = false;
  if (this.x < -margin || this.x > W + margin || this.y < -margin || this.y > H + margin) {
    exited = true;
  }
  // Also reset if too far from center after crossing (safety)
  if (this.inIntersection && distToCenter > Math.max(W, H) * 0.8) {
    exited = true;
  }

  if (exited) this.reset(false);
};

Particle.prototype.draw = function(ctx) {
  // Trail with direction-color
  if (this.trail.length > 1) {
    ctx.beginPath();
    ctx.moveTo(this.trail[0].x, this.trail[0].y);
    for (var i = 1; i < this.trail.length; i++) {
      ctx.lineTo(this.trail[i].x, this.trail[i].y);
    }
    var ta = this.alpha * (this.colorTransition > 0.3 ? 0.4 : 0.2);
    ctx.strokeStyle = 'rgba(' + CFG.colors.trail.join(',') + ',' + ta + ')';
    ctx.lineWidth = this.radius * 0.6;
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  // Glow
  var gg = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, CFG.glowRadius);
  gg.addColorStop(0, 'rgba(' + this.color.join(',') + ',' + this.alpha + ')');
  gg.addColorStop(1, 'rgba(' + this.color.join(',') + ',0)');
  ctx.beginPath();
  ctx.arc(this.x, this.y, CFG.glowRadius, 0, Math.PI*2);
  ctx.fillStyle = gg;
  ctx.fill();

  // Core dot
  ctx.beginPath();
  ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(' + this.color.join(',') + ',' + Math.min(1, this.alpha + 0.15) + ')';
  ctx.fill();
};

// ── Draw road infrastructure ──────────────────────────────────

function drawRoads() {
  var cx = W/2, cy = H/2;
  var roadLen = Math.max(W, H);

  ctx.save();
  ctx.globalAlpha = 0.03;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.setLineDash([20, 30]);

  // Horizontal road (E↔W)
  for (var lane = -2; lane <= 2; lane++) {
    var offset = lane * 22;
    ctx.beginPath();
    ctx.moveTo(-20, cy + offset);
    ctx.lineTo(W + 20, cy + offset);
    ctx.stroke();
  }

  // Vertical road (N↔S)
  for (var lane = -2; lane <= 2; lane++) {
    var offset = lane * 22;
    ctx.beginPath();
    ctx.moveTo(cx + offset, -20);
    ctx.lineTo(cx + offset, H + 20);
    ctx.stroke();
  }

  ctx.setLineDash([]);

  // Center intersection circle
  ctx.beginPath();
  ctx.arc(cx, cy, 48, 0, Math.PI*2);
  ctx.globalAlpha = 0.04;
  ctx.fillStyle = '#5b9bd5';
  ctx.fill();
  ctx.globalAlpha = 0.06;
  ctx.strokeStyle = '#5b9bd5';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Stop lines (subtle)
  var stopDist = 55;
  ctx.globalAlpha = 0.04;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.5;
  // North stop line
  ctx.beginPath(); ctx.moveTo(cx - 50, cy - stopDist); ctx.lineTo(cx + 50, cy - stopDist); ctx.stroke();
  // South stop line
  ctx.beginPath(); ctx.moveTo(cx - 50, cy + stopDist); ctx.lineTo(cx + 50, cy + stopDist); ctx.stroke();
  // East stop line
  ctx.beginPath(); ctx.moveTo(cx + stopDist, cy - 50); ctx.lineTo(cx + stopDist, cy + 50); ctx.stroke();
  // West stop line
  ctx.beginPath(); ctx.moveTo(cx - stopDist, cy - 50); ctx.lineTo(cx - stopDist, cy + 50); ctx.stroke();

  ctx.restore();
}

// ── Connection lines between nearby particles ────────────────

function drawConnections() {
  for (var i = 0; i < particles.length; i++) {
    for (var j = i + 1; j < particles.length; j++) {
      var dx = particles[i].x - particles[j].x;
      var dy = particles[i].y - particles[j].y;
      var dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < CFG.connectionDist) {
        var alpha = (1 - dist/CFG.connectionDist) * 0.08;
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.strokeStyle = 'rgba(91,155,213,' + alpha + ')';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }
}

// ── Mouse conflict zone overlay ──────────────────────────────

function drawMouseZone() {
  if (!mouse.active) return;

  // Pulsing ring
  var pulse = 0.5 + 0.5 * Math.sin(frame * 0.025);
  var grad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, CFG.mouseRadius);
  grad.addColorStop(0, 'rgba(213,91,91,' + (0.05 * pulse) + ')');
  grad.addColorStop(0.4, 'rgba(212,149,107,0.03)');
  grad.addColorStop(1, 'rgba(226,185,111,0)');

  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, CFG.mouseRadius, 0, Math.PI*2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Outer ring
  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, CFG.mouseRadius * (0.45 + 0.55 * pulse), 0, Math.PI*2);
  ctx.strokeStyle = 'rgba(213,91,91,' + (0.1 * pulse) + ')';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ── Resize ────────────────────────────────────────────────────

function resize() {
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight;
  var target = Math.floor((W * H) / 10000);
  var count = Math.min(220, Math.max(80, target));
  while (particles.length < count) particles.push(new Particle());
  while (particles.length > count) particles.pop();
}

// ── Event handlers ────────────────────────────────────────────

function onMouseMove(e) { mouse.x = e.clientX; mouse.y = e.clientY; mouse.active = true; }
function onMouseLeave() { mouse.active = false; }
function onTouchMove(e) { if (e.touches.length > 0) { mouse.x = e.touches[0].clientX; mouse.y = e.touches[0].clientY; mouse.active = true; } }
function onTouchEnd() { mouse.active = false; }

// ── Animation loop ────────────────────────────────────────────

function animate() {
  // Subtle trail fade
  ctx.fillStyle = 'rgba(13,13,15,0.12)';
  ctx.fillRect(0, 0, W, H);

  drawRoads();
  drawConnections();

  for (var i = 0; i < particles.length; i++) {
    particles[i].update();
    particles[i].draw(ctx);
  }

  drawMouseZone();
  frame++;
  requestAnimationFrame(animate);
}

// ── Init ──────────────────────────────────────────────────────

window.addEventListener('resize', resize);
window.addEventListener('mousemove', onMouseMove);
window.addEventListener('mouseleave', onMouseLeave);
window.addEventListener('touchmove', onTouchMove, {passive:true});
window.addEventListener('touchend', onTouchEnd);

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() { resize(); animate(); });
} else {
  resize(); animate();
}
})();
