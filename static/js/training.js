// Training playback page — animated loss & accuracy charts from recorded run
(function () {
  const $ = (id) => document.getElementById(id);
  const lossC = $("loss-chart");
  const accC = $("acc-chart");
  const playBtn = $("play");
  const resetBtn = $("reset");
  const tStatus = $("train-status");
  const tbody = document.querySelector("#epoch-table tbody");

  let history = null;
  let playing = false;
  let step = 0;

  async function load() {
    const resp = await fetch("/api/training-history");
    if (!resp.ok) {
      tStatus.textContent = "no training history available — run train.py";
      return;
    }
    history = await resp.json();
    drawCharts(0);
    fillTable();
    tStatus.textContent = `loaded ${history.batches.length} batch samples / ${history.epochs.length} epochs`;
  }

  function fillTable() {
    tbody.innerHTML = "";
    for (const e of history.epochs) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${e.epoch + 1}</td><td>${e.loss.toFixed(4)}</td>
        <td>${(e.accuracy*100).toFixed(2)}%</td>
        <td>${e.val_loss.toFixed(4)}</td>
        <td>${(e.val_accuracy*100).toFixed(2)}%</td>`;
      tbody.appendChild(tr);
    }
  }

  function drawCharts(cutoff) {
    drawLoss(cutoff);
    drawAcc(cutoff);
  }

  function drawAxis(ctx, w, h, xLabel, yLabel) {
    ctx.fillStyle = "#04060a";
    ctx.fillRect(0, 0, w, h);
    // grid
    ctx.strokeStyle = "rgba(120,170,220,0.12)";
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 5; i++) {
      const y = (h / 5) * i;
      ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(w - 10, y); ctx.stroke();
    }
    // axes
    ctx.strokeStyle = "#1d2a3d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 24);
    ctx.lineTo(w - 10, h - 24);
    ctx.stroke();
    ctx.fillStyle = "#6f829a";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText(xLabel, w - 60, h - 8);
    ctx.save();
    ctx.translate(12, 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  function drawSeries(ctx, w, h, data, xKey, yKey, color, max, glow) {
    if (data.length === 0) return;
    const padL = 40, padR = 10, padT = 10, padB = 24;
    const xMax = data[data.length - 1][xKey] || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    if (glow) ctx.shadowBlur = 10, ctx.shadowColor = color;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = padL + ((d[xKey] / xMax) * (w - padL - padR));
      const y = (h - padB) - ((d[yKey] / max) * (h - padT - padB));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function drawLoss(cutoff) {
    const ctx = lossC.getContext("2d");
    const w = lossC.width, h = lossC.height;
    drawAxis(ctx, w, h, "step", "loss");
    if (!history) return;
    // pseudo-x: flatten batches over a synthetic counter
    const batches = history.batches.map((b, i) => ({ x: i, loss: b.loss }));
    const epochs = history.epochs.map((e, i) => ({
      x: ((i + 1) / history.epochs.length) * batches.length,
      loss: e.val_loss,
    }));
    const max = Math.max(...batches.map((b) => b.loss), ...epochs.map((e) => e.loss)) * 1.05 || 2.0;
    const slice = batches.slice(0, cutoff);
    const epSlice = epochs.filter((e) => e.x <= (cutoff || 0));
    drawSeries(ctx, w, h, slice, "x", "loss", "#00f0ff", max, true);
    drawSeries(ctx, w, h, epSlice, "x", "loss", "#ff2bd6", max, true);
    legend(ctx, w, [["train loss", "#00f0ff"], ["val loss", "#ff2bd6"]]);
  }

  function drawAcc(cutoff) {
    const ctx = accC.getContext("2d");
    const w = accC.width, h = accC.height;
    drawAxis(ctx, w, h, "step", "acc");
    if (!history) return;
    const batches = history.batches.map((b, i) => ({ x: i, acc: b.accuracy }));
    const epochs = history.epochs.map((e, i) => ({
      x: ((i + 1) / history.epochs.length) * batches.length,
      acc: e.val_accuracy,
    }));
    const slice = batches.slice(0, cutoff);
    const epSlice = epochs.filter((e) => e.x <= (cutoff || 0));
    drawSeries(ctx, w, h, slice, "x", "acc", "#29ffa6", 1.0, true);
    drawSeries(ctx, w, h, epSlice, "x", "acc", "#ffb547", 1.0, true);
    legend(ctx, w, [["train acc", "#29ffa6"], ["val acc", "#ffb547"]]);
  }

  function legend(ctx, w, items) {
    ctx.font = "10px 'JetBrains Mono', monospace";
    let x = w - 200;
    items.forEach(([label, color]) => {
      ctx.fillStyle = color;
      ctx.fillRect(x, 14, 10, 4);
      ctx.fillStyle = "#d6e4f5";
      ctx.fillText(label, x + 14, 20);
      x += 90;
    });
  }

  playBtn.addEventListener("click", () => {
    if (playing) return;
    if (!history) return;
    playing = true;
    if (step >= history.batches.length) step = 0;
    const total = history.batches.length;
    const interval = Math.max(8, Math.floor(8000 / total));
    function tick() {
      if (!playing) return;
      step = Math.min(total, step + Math.max(1, Math.floor(total / 200)));
      drawCharts(step);
      tStatus.textContent = `step ${step}/${total}`;
      if (step >= total) { playing = false; tStatus.textContent = "done"; return; }
      setTimeout(tick, interval);
    }
    tick();
  });

  resetBtn.addEventListener("click", () => {
    playing = false; step = 0; drawCharts(0); tStatus.textContent = "reset";
  });

  load();
})();
