// Particle/grid background effect — a faint moving neural-net field
(function() {
  const canvas = document.getElementById("bgfx");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let w, h, nodes;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    const count = Math.floor((w * h) / 22000);
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
    }));
  }
  resize();
  window.addEventListener("resize", resize);

  function frame() {
    ctx.clearRect(0, 0, w, h);
    // links
    ctx.lineWidth = 1;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      a.x += a.vx; a.y += a.vy;
      if (a.x < 0 || a.x > w) a.vx *= -1;
      if (a.y < 0 || a.y > h) a.vy *= -1;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 18000) {
          const alpha = 1 - d2 / 18000;
          ctx.strokeStyle = `rgba(40, 214, 0, ${alpha * 0.22})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    // dots
    ctx.fillStyle = "rgba(40, 214, 0, 0.65)";
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }
  frame();
})();
