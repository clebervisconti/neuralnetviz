// Inference page: network SVG + animated activation flow + heatmaps
(function () {
  const $ = (id) => document.getElementById(id);
  const dz = $("dropzone");
  const file = $("file");
  const preview = $("preview");
  const status = $("status");
  const predictionsEl = $("predictions");
  const fmEl = $("featuremaps");
  const svg = $("network");
  const samples = $("samples");
  const modeToggle = $("mode-toggle");
  const imagenetTitle = $("imagenet-title");
  const imagenetEl = $("imagenet-top");
  const tooltip = $("node-tooltip");

  const SVG_NS = "http://www.w3.org/2000/svg";

  let arch = null;
  let nodes = []; // {layer, x, y, type}
  let mode = "teaching";
  let lastUploadedBlob = null;
  let lastPrediction = null; // {predictions, layers}

  // ----- color per layer type (CV brand) -----
  // Verde Ascensão #28d600 is the accent; dim shades distinguish other types
  // without introducing new hues (brand rule: single accent).
  const TYPE_COLOR = {
    InputLayer: "#ffffff",
    Conv2D: "#28d600",
    MaxPooling2D: "rgba(40,214,0,0.55)",
    GlobalAveragePooling2D: "rgba(40,214,0,0.55)",
    Dense: "rgba(255,255,255,0.85)",
  };
  function colorFor(layer, isOutput) {
    if (isOutput) return "#28d600";
    return TYPE_COLOR[layer.type] || "#ffffff";
  }

  // ----- build the SVG network diagram -----
  async function buildNetwork() {
    const resp = await fetch(`/api/architecture?mode=${mode}`);
    arch = await resp.json();
    drawNetwork(arch);
  }

  function drawNetwork(arch) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const W = 900, H = 540;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    // defs: gradient + glow filter
    const defs = el("defs");
    defs.innerHTML = `
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <linearGradient id="edge" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#28d600" stop-opacity="0.0"/>
        <stop offset="50%" stop-color="#28d600" stop-opacity="0.95"/>
        <stop offset="100%" stop-color="#28d600" stop-opacity="0.0"/>
      </linearGradient>`;
    svg.appendChild(defs);

    const layers = arch.layers;
    const N = layers.length;
    const padX = 70;
    const colW = (W - padX * 2) / Math.max(1, N - 1);

    nodes = layers.map((layer, i) => {
      const isOutput = i === N - 1;
      const isInput = i === 0;
      // node count visualized per layer: input is single image, output is class neurons,
      // dense layers show a few neurons, conv layers show a few channels stacked vertically
      let count = 1;
      if (isInput) count = 1;
      else if (layer.type === "Dense") count = isOutput ? arch.classes.length : 6;
      else if (layer.type === "Conv2D" || layer.type === "MaxPooling2D") count = 5;
      else count = 3;

      const x = padX + i * colW;
      const cy = H / 2;
      const spread = Math.min(380, count * 36);
      const positions = [];
      for (let k = 0; k < count; k++) {
        const y = cy + ((k - (count - 1) / 2) * (spread / Math.max(1, count - 1) || 0));
        positions.push({ x, y });
      }
      return { layer, positions, isOutput, isInput };
    });

    // edges
    for (let i = 0; i < nodes.length - 1; i++) {
      const a = nodes[i], b = nodes[i + 1];
      for (const pa of a.positions) {
        for (const pb of b.positions) {
          const line = el("line", {
            x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y,
            stroke: "rgba(120, 170, 220, 0.10)",
            "stroke-width": 0.8,
            class: `edge edge-${i}-${i + 1}`,
          });
          svg.appendChild(line);
        }
      }
    }

    // nodes
    nodes.forEach((n, i) => {
      const c = colorFor(n.layer, n.isOutput);
      n.positions.forEach((p, pi) => {
        const circle = el("circle", {
          cx: p.x, cy: p.y, r: 7,
          fill: "#1e1e1e",
          stroke: c, "stroke-width": 1.8,
          filter: "url(#glow)",
          class: `node node-${i}`,
        });
        circle.dataset.baseColor = c;
        circle.dataset.layerIdx = String(i);
        circle.dataset.posIdx = String(pi);
        circle.addEventListener("mouseenter", onNodeHover);
        circle.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(circle);
      });
      // layer label
      const labelType = n.layer.type.replace("2D", "");
      const labelShape = Array.isArray(n.layer.output_shape)
        ? n.layer.output_shape.slice(1).filter((d) => d !== null && d !== "?").join("×")
        : "";
      const label = el("text", {
        x: n.positions[0].x,
        y: 30,
        "text-anchor": "middle",
        fill: c,
        "font-size": 10,
        "font-family": "JetBrains Mono, monospace",
        "letter-spacing": "0.18em",
      });
      label.textContent = labelType.toUpperCase();
      svg.appendChild(label);

      const shapeLabel = el("text", {
        x: n.positions[0].x,
        y: 46,
        "text-anchor": "middle",
        fill: "rgba(255,255,255,0.45)",
        "font-size": 9,
        "font-family": "JetBrains Mono, monospace",
      });
      shapeLabel.textContent = labelShape;
      svg.appendChild(shapeLabel);
    });

    // output class labels
    const out = nodes[nodes.length - 1];
    out.positions.forEach((p, k) => {
      const t = el("text", {
        x: p.x + 14,
        y: p.y + 4,
        fill: "#ffffff",
        "font-size": 11,
        "font-family": "JetBrains Mono, monospace",
        "letter-spacing": "0.1em",
        class: "out-label",
        "data-idx": k,
      });
      t.textContent = arch.classes[k];
      svg.appendChild(t);
    });

    // input thumbnail placeholder
    const inp = nodes[0];
    const ip = inp.positions[0];
    const rect = el("rect", {
      x: ip.x - 22, y: ip.y - 22, width: 44, height: 44,
      fill: "rgba(40,214,0,0.05)",
      stroke: "#28d600", "stroke-width": 1.4,
      rx: 4, ry: 4,
      filter: "url(#glow)",
      class: "input-box",
    });
    svg.appendChild(rect);
  }

  function el(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // ----- animation: pulse data along edges between layer i and i+1 -----
  function pulseBetween(i, durationMs) {
    return new Promise((resolve) => {
      const edges = svg.querySelectorAll(`.edge-${i}-${i + 1}`);
      edges.forEach((e) => {
        e.setAttribute("stroke", "url(#edge)");
        e.setAttribute("stroke-width", 1.4);
        e.style.opacity = 0.9;
      });
      // light up next layer's nodes
      const nextNodes = svg.querySelectorAll(`.node-${i + 1}`);
      nextNodes.forEach((n) => {
        const c = n.dataset.baseColor;
        n.setAttribute("fill", c);
        n.style.transition = "fill 0.4s";
      });
      setTimeout(() => {
        edges.forEach((e) => {
          e.setAttribute("stroke", "rgba(120, 170, 220, 0.10)");
          e.setAttribute("stroke-width", 0.8);
          e.style.opacity = 1;
        });
        resolve();
      }, durationMs);
    });
  }

  async function animateForward() {
    // reset
    svg.querySelectorAll(".node").forEach((n) => n.setAttribute("fill", "#0a1018"));
    // light the input box
    const input = svg.querySelector(".input-box");
    if (input) {
      input.setAttribute("fill", "rgba(40,214,0,0.25)");
      setTimeout(() => input.setAttribute("fill", "rgba(40,214,0,0.05)"), 400);
    }
    for (let i = 0; i < nodes.length - 1; i++) {
      await pulseBetween(i, 320);
    }
  }

  // ----- heatmap rendering -----
  function renderFeatureMaps(layerSummaries) {
    fmEl.innerHTML = "";
    layerSummaries.forEach((layer) => {
      if (!layer.heatmaps && !layer.values) return;
      const group = document.createElement("div");
      group.className = "fm-group";
      group.innerHTML = `<h4>${layer.name} <span class="muted">${layer.shape.slice(1).filter(d=>d!==null).join("×")}</span></h4>`;
      const row = document.createElement("div");
      row.className = "fm-row";

      if (layer.heatmaps) {
        for (const hm of layer.heatmaps) {
          const c = document.createElement("canvas");
          c.width = hm.w; c.height = hm.h;
          const ctx = c.getContext("2d");
          const img = ctx.createImageData(hm.w, hm.h);
          for (let i = 0; i < hm.values.length; i++) {
            const t = hm.values[i] / 255;
            // dark Preto Neural -> Verde Ascensão ramp
            img.data[i * 4 + 0] = Math.round(0x1e + (0x28 - 0x1e) * t);
            img.data[i * 4 + 1] = Math.round(0x1e + (0xd6 - 0x1e) * t);
            img.data[i * 4 + 2] = Math.round(0x1e + (0x00 - 0x1e) * t);
            img.data[i * 4 + 3] = 235;
          }
          ctx.putImageData(img, 0, 0);
          row.appendChild(c);
        }
      } else if (layer.values) {
        const max = Math.max(1e-6, ...layer.values.map(Math.abs));
        const c = document.createElement("canvas");
        c.width = layer.values.length;
        c.height = 1;
        c.classList.add("dense");
        const ctx = c.getContext("2d");
        const img = ctx.createImageData(c.width, 1);
        layer.values.forEach((v, i) => {
          const t = Math.max(0, Math.min(1, v / max));
          img.data[i * 4 + 0] = Math.round(0x1e + (0x28 - 0x1e) * t);
          img.data[i * 4 + 1] = Math.round(0x1e + (0xd6 - 0x1e) * t);
          img.data[i * 4 + 2] = Math.round(0x1e + (0x00 - 0x1e) * t);
          img.data[i * 4 + 3] = 235;
        });
        ctx.putImageData(img, 0, 0);
        row.appendChild(c);
      }
      group.appendChild(row);
      fmEl.appendChild(group);
    });
  }

  // ----- predictions list -----
  function renderPredictions(preds) {
    predictionsEl.innerHTML = "";
    preds.forEach((p, i) => {
      const li = document.createElement("li");
      if (i === 0) li.classList.add("top");
      const pct = (p.prob * 100).toFixed(1);
      li.innerHTML = `
        <span class="label">${p.label}</span>
        <span class="bar"><span style="width:${pct}%"></span></span>
        <span class="pct">${pct}%</span>`;
      predictionsEl.appendChild(li);
    });

    // also flash the output node of the top class
    const top = preds[0];
    const topIdx = arch.classes.indexOf(top.label);
    const outNode = svg.querySelectorAll(`.node-${nodes.length - 1}`)[topIdx];
    if (outNode) {
      outNode.setAttribute("fill", "#28d600");
      outNode.setAttribute("r", 10);
    }
    // highlight output text labels
    svg.querySelectorAll(".out-label").forEach((t) => {
      const idx = parseInt(t.dataset.idx, 10);
      if (idx === topIdx) {
        t.setAttribute("fill", "#28d600");
        t.setAttribute("font-size", 13);
      } else {
        t.setAttribute("fill", "rgba(255,255,255,0.45)");
        t.setAttribute("font-size", 11);
      }
    });
  }

  // ----- prediction request -----
  async function predict(blob) {
    setStatus("encoding image…", "busy");
    const fd = new FormData();
    fd.append("image", blob, "input.png");

    setStatus("running forward pass…", "busy");
    const animPromise = animateForward();
    const resp = await fetch(`/api/predict?mode=${mode}`, { method: "POST", body: fd });
    if (!resp.ok) {
      const t = await resp.text();
      setStatus(`error: ${t}`, "err");
      return;
    }
    const data = await resp.json();
    await animPromise;

    lastPrediction = data;
    if (data.preview) {
      preview.src = data.preview;
      dz.classList.add("has-image");
    }
    renderPredictions(data.predictions);
    renderFeatureMaps(data.layers);
    renderImagenetTop(data.imagenet_top);
    hideTooltip();
    setStatus(`predicted ${data.predictions[0].label} (${(data.predictions[0].prob*100).toFixed(1)}%) · ${mode}`, "ok");
  }

  function setStatus(msg, cls) {
    status.textContent = msg;
    status.className = "status" + (cls ? " " + cls : "");
  }

  // ----- file/drag handlers -----
  dz.addEventListener("click", () => file.click());
  file.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
  ["dragenter","dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add("drag");
  }));
  ["dragleave","drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.remove("drag");
  }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  function handleFile(f) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.src = ev.target.result;
      dz.classList.add("has-image");
    };
    reader.readAsDataURL(f);
    lastUploadedBlob = f;
    predict(f);
  }

  function renderImagenetTop(top) {
    if (!top || !top.length) {
      imagenetTitle.style.display = "none";
      imagenetEl.innerHTML = "";
      return;
    }
    imagenetTitle.style.display = "block";
    imagenetEl.innerHTML = "";
    top.forEach((p) => {
      const li = document.createElement("li");
      const pct = (p.prob * 100).toFixed(1);
      const pretty = p.label.replace(/_/g, " ");
      li.innerHTML = `<span class="label">${pretty}</span><span class="pct">${pct}%</span>`;
      imagenetEl.appendChild(li);
    });
  }

  // ----- mode toggle -----
  async function switchMode(newMode) {
    if (newMode === mode) return;
    mode = newMode;
    modeToggle.querySelectorAll(".mode-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === mode);
      b.setAttribute("aria-selected", b.dataset.mode === mode ? "true" : "false");
      b.disabled = true;
    });
    setStatus(`loading ${mode} model…`, "busy");
    try {
      await buildNetwork();
      setStatus(`${mode} model ready · ${arch.input_size}×${arch.input_size} input`, "ok");
      if (lastUploadedBlob) await predict(lastUploadedBlob);
    } catch (e) {
      setStatus(`failed to switch mode: ${e.message}`, "err");
    } finally {
      modeToggle.querySelectorAll(".mode-opt").forEach((b) => (b.disabled = false));
    }
  }
  modeToggle.querySelectorAll(".mode-opt").forEach((b) => {
    b.addEventListener("click", () => switchMode(b.dataset.mode));
  });

  // ----- sample buttons -----
  const SAMPLES = ["bird","cat","deer","dog","frog","horse"];
  function buildSamples() {
    SAMPLES.forEach((label) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.addEventListener("click", async () => {
        setStatus(`loading sample ${label}…`, "busy");
        try {
          const resp = await fetch(`/static/img/${label}.jpg`);
          if (!resp.ok) {
            setStatus(`no sample image for ${label}`, "err");
            return;
          }
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          preview.src = url;
          dz.classList.add("has-image");
          predict(blob);
        } catch (err) {
          setStatus(`sample load failed: ${err.message}`, "err");
        }
      });
      samples.appendChild(b);
    });
  }

  // ----- hover tooltip on network nodes -----
  function onNodeHover(ev) {
    const circle = ev.currentTarget;
    const layerIdx = parseInt(circle.dataset.layerIdx, 10);
    const posIdx = parseInt(circle.dataset.posIdx, 10);
    const n = nodes[layerIdx];
    if (!n) return;
    showTooltip(circle, layerIdx, posIdx, n);
  }

  function fmt(x, digits) {
    if (x === undefined || x === null || Number.isNaN(x)) return "—";
    return Number(x).toFixed(digits);
  }

  function showTooltip(circle, layerIdx, posIdx, n) {
    const wrap = $("network-wrap");
    const wrapRect = wrap.getBoundingClientRect();
    const r = circle.getBoundingClientRect();
    const x = r.left - wrapRect.left + r.width / 2;
    const y = r.top - wrapRect.top;

    let html = "";
    if (n.isOutput && lastPrediction && arch) {
      // Each output dot is one class. The pos index maps directly to the
      // class index because we drew the output positions in class order.
      const classes = arch.classes;
      const className = classes[posIdx] || `class ${posIdx}`;
      const ranked = lastPrediction.predictions || [];
      const entry = ranked.find((p) => p.label === className);
      const prob = entry ? entry.prob : 0;
      const pct = (prob * 100).toFixed(1);
      const rank = ranked.findIndex((p) => p.label === className) + 1;
      const rankStr = rank > 0 ? `#${rank} of ${ranked.length}` : "—";
      html = `
        <div class="nt-title">Output · ${className}</div>
        <div class="nt-row"><span>probability</span><b>${pct}%</b></div>
        <div class="nt-row"><span>rank</span><b>${rankStr}</b></div>
        <div class="nt-bar"><span style="width:${pct}%"></span></div>
        <div class="nt-hint">softmax score — confidence the input is "${className}"</div>`;
    } else if (n.isInput) {
      const size = (arch && arch.input_size) || "?";
      html = `
        <div class="nt-title">Input · ${n.layer.name}</div>
        <div class="nt-row"><span>shape</span><b>${size}×${size}×3</b></div>
        <div class="nt-hint">your image is resized to this size and normalized before the model sees it</div>`;
    } else {
      // Hidden layer — pull from lastPrediction.layers if available.
      const layerName = n.layer.name;
      const layerStats = (lastPrediction && lastPrediction.layers)
        ? lastPrediction.layers.find((l) => l.name === layerName)
        : null;
      const shape = n.layer.output_shape
        ? n.layer.output_shape.slice(1).filter((d) => d !== null && d !== "?").join("×")
        : "—";
      let body = `
        <div class="nt-row"><span>shape</span><b>${shape}</b></div>`;
      if (layerStats) {
        const mean = fmt(layerStats.mean, 3);
        const max = fmt(layerStats.max, 3);
        const min = fmt(layerStats.min, 3);
        const topCh = layerStats.heatmaps && layerStats.heatmaps[0]
          ? `ch ${layerStats.heatmaps[0].channel}`
          : (layerStats.values ? `unit ${argmax(layerStats.values)}` : "—");
        // map the dot's position index to a sample channel/unit for richer info
        let perDot = "";
        if (layerStats.heatmaps && layerStats.heatmaps[posIdx]) {
          const hm = layerStats.heatmaps[posIdx];
          const peak = Math.max(...hm.values) / 255;
          perDot = `<div class="nt-row"><span>this node · ch ${hm.channel}</span><b>peak ${fmt(peak, 2)}</b></div>`;
        } else if (layerStats.values && layerStats.values.length > posIdx) {
          perDot = `<div class="nt-row"><span>this node · unit ${posIdx}</span><b>${fmt(layerStats.values[posIdx], 3)}</b></div>`;
        }
        body += `
          ${perDot}
          <div class="nt-row"><span>mean act.</span><b>${mean}</b></div>
          <div class="nt-row"><span>max act.</span><b>${max}</b></div>
          <div class="nt-row"><span>min act.</span><b>${min}</b></div>
          <div class="nt-row"><span>top</span><b>${topCh}</b></div>`;
      } else {
        body += `<div class="nt-hint">run a prediction to see activation scores</div>`;
      }
      html = `<div class="nt-title">${n.layer.type.replace("2D", "")} · ${layerName}</div>${body}`;
    }

    tooltip.innerHTML = html;
    // position; clamp so it stays inside the wrap
    const maxX = wrapRect.width - 20;
    tooltip.style.left = `${Math.min(Math.max(20, x), maxX)}px`;
    tooltip.style.top = `${Math.max(8, y)}px`;
    tooltip.setAttribute("data-visible", "true");
    tooltip.setAttribute("aria-hidden", "false");
  }

  function argmax(arr) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
    return bi;
  }

  function hideTooltip() {
    if (!tooltip) return;
    tooltip.setAttribute("data-visible", "false");
    tooltip.setAttribute("aria-hidden", "true");
  }

  // ----- init -----
  buildNetwork().then(() => {
    buildSamples();
    setStatus("ready — upload an image to begin");
  }).catch((e) => {
    setStatus(`failed to load architecture: ${e.message}`, "err");
  });
})();
