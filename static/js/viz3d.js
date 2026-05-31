// 3D network visualization using Three.js
//
// Layers are spread along the X-axis; within each layer the visible "nodes"
// are arranged in a vertical column (matching the 2D dot count per layer).
// Edges are thin lines between adjacent layers. After each prediction, each
// sphere's emissive intensity and a billboard sprite next to it reflect the
// per-node score we already pipe through paintNodeScores in viz.js.
//
// Exposes a tiny imperative API on window.NNV3D so viz.js can drive it
// without importing.

(function () {
  if (typeof THREE === "undefined") {
    console.warn("[NNV3D] Three.js not loaded — 3D view disabled");
    window.NNV3D = { mount: () => {}, updateScores: () => {}, pause: () => {} };
    return;
  }

  const ACCENT = 0x28d600;
  const WHITE = 0xffffff;
  const DIM_WHITE = 0xa0a0a0;
  const POOL_GREEN = 0x19a300;

  let scene = null, camera = null, renderer = null, controls = null;
  let raf = null;
  let nodeMeshes = []; // [[meshA, meshB, ...] per layer]
  let labelSprites = []; // matching shape — billboard labels per node
  let edgeLines = [];
  let layerLabelSprites = [];
  let containerEl = null;
  let resizeObs = null;
  let archCache = null;

  function colorForType(type, isOutput) {
    if (isOutput) return ACCENT;
    if (type === "Conv2D") return ACCENT;
    if (type === "MaxPooling2D" || type === "GlobalAveragePooling2D") return POOL_GREEN;
    if (type === "Dense") return WHITE;
    return WHITE; // input
  }

  function makeLabelTexture(text, color = "#ffffff", bg = "rgba(20,20,20,0.85)") {
    const cvs = document.createElement("canvas");
    cvs.width = 256; cvs.height = 96;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    ctx.fillStyle = bg;
    roundRect(ctx, 4, 12, cvs.width - 8, cvs.height - 24, 14);
    ctx.fill();
    ctx.strokeStyle = "rgba(40,214,0,0.7)";
    ctx.lineWidth = 2;
    roundRect(ctx, 4, 12, cvs.width - 8, cvs.height - 24, 14);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "bold 48px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cvs.width / 2, cvs.height / 2);
    const tex = new THREE.CanvasTexture(cvs);
    tex.anisotropy = 4;
    return tex;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function setLabel(sprite, text, color) {
    sprite.material.map = makeLabelTexture(text, color);
    sprite.material.needsUpdate = true;
  }

  function clearScene() {
    if (!scene) return;
    nodeMeshes.flat().forEach((m) => scene.remove(m));
    edgeLines.forEach((l) => scene.remove(l));
    labelSprites.flat().forEach((s) => scene.remove(s));
    layerLabelSprites.forEach((s) => scene.remove(s));
    nodeMeshes = []; edgeLines = []; labelSprites = []; layerLabelSprites = [];
  }

  function buildScene(arch) {
    archCache = arch;
    clearScene();

    const layers = arch.layers;
    const N = layers.length;
    const spanX = 14;
    const stepX = N > 1 ? spanX / (N - 1) : 0;
    const xs = layers.map((_, i) => -spanX / 2 + i * stepX);

    layers.forEach((layer, i) => {
      const isOutput = i === N - 1;
      const isInput = i === 0;
      let count = 1;
      if (isInput) count = 1;
      else if (layer.type === "Dense") count = isOutput ? arch.classes.length : 6;
      else if (layer.type === "Conv2D" || layer.type === "MaxPooling2D") count = 5;
      else count = 3;

      const spread = Math.min(7.5, count * 1.2);
      const positions = [];
      for (let k = 0; k < count; k++) {
        const y = (k - (count - 1) / 2) * (count > 1 ? spread / (count - 1) : 0);
        positions.push([xs[i], y, 0]);
      }

      const color = colorForType(layer.type, isOutput);
      const meshes = [];
      const labels = [];
      positions.forEach((p, pi) => {
        const geom = new THREE.SphereGeometry(0.42, 32, 24);
        const mat = new THREE.MeshStandardMaterial({
          color: 0x1e1e1e,
          emissive: color,
          emissiveIntensity: 0.15,
          metalness: 0.3,
          roughness: 0.4,
        });
        const sphere = new THREE.Mesh(geom, mat);
        sphere.position.set(p[0], p[1], p[2]);
        sphere.userData = { layerIdx: i, posIdx: pi, baseColor: color };
        scene.add(sphere);
        meshes.push(sphere);

        // billboard label sprite
        const tex = makeLabelTexture(isInput ? "IN" : "—", "#cccccc");
        const sprMat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
        const sprite = new THREE.Sprite(sprMat);
        sprite.position.set(p[0], p[1] - 0.85, p[2]);
        sprite.scale.set(1.4, 0.55, 1);
        scene.add(sprite);
        labels.push(sprite);
      });
      nodeMeshes.push(meshes);
      labelSprites.push(labels);

      // layer type label above the column
      const lt = makeLabelTexture(layer.type.replace("2D", "").toUpperCase(),
        "#" + color.toString(16).padStart(6, "0"));
      const lsMat = new THREE.SpriteMaterial({ map: lt, transparent: true, depthTest: false });
      const ls = new THREE.Sprite(lsMat);
      ls.position.set(xs[i], spread / 2 + 1.2, 0);
      ls.scale.set(1.8, 0.7, 1);
      scene.add(ls);
      layerLabelSprites.push(ls);
    });

    // edges
    const edgeMat = new THREE.LineBasicMaterial({
      color: ACCENT, transparent: true, opacity: 0.12,
    });
    for (let i = 0; i < nodeMeshes.length - 1; i++) {
      for (const a of nodeMeshes[i]) {
        for (const b of nodeMeshes[i + 1]) {
          const geom = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
          const line = new THREE.Line(geom, edgeMat);
          scene.add(line);
          edgeLines.push(line);
        }
      }
    }
  }

  function ensureRendererSized() {
    if (!containerEl || !renderer) return;
    const w = containerEl.clientWidth || 600;
    const h = containerEl.clientHeight || 480;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function loop() {
    if (!renderer || !scene || !camera) return;
    raf = requestAnimationFrame(loop);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  function pause() {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
  }

  function mount(containerId, arch) {
    containerEl = document.getElementById(containerId);
    if (!containerEl || !arch) return;

    if (!renderer) {
      scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x06090d, 18, 36);

      camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
      camera.position.set(0, 2, 16);
      camera.lookAt(0, 0, 0);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x000000, 0);
      containerEl.appendChild(renderer.domElement);

      const ambient = new THREE.AmbientLight(0xffffff, 0.55);
      scene.add(ambient);
      const key = new THREE.DirectionalLight(0xffffff, 0.55);
      key.position.set(6, 8, 10);
      scene.add(key);
      const rim = new THREE.PointLight(ACCENT, 1.0, 30);
      rim.position.set(0, 0, 8);
      scene.add(rim);

      if (THREE.OrbitControls) {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.rotateSpeed = 0.7;
        controls.zoomSpeed = 0.6;
        controls.target.set(0, 0, 0);
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.6;
      }
      resizeObs = new ResizeObserver(ensureRendererSized);
      resizeObs.observe(containerEl);
    }

    buildScene(arch);
    ensureRendererSized();
    if (!raf) loop();
  }

  function updateScores(nodes, data, arch) {
    if (!scene || !nodes.length || !nodeMeshes.length) return;
    archCache = arch || archCache;
    const layerByName = {};
    (data.layers || []).forEach((l) => (layerByName[l.name] = l));

    nodes.forEach((n, i) => {
      const meshes = nodeMeshes[i] || [];
      const labels = labelSprites[i] || [];
      const stats = layerByName[n.layer.name];
      meshes.forEach((mesh, pi) => {
        let text = "—", lit = 0;
        if (n.isOutput && archCache) {
          const cls = archCache.classes[pi];
          const entry = (data.predictions || []).find((p) => p.label === cls);
          const prob = entry ? entry.prob : 0;
          text = `${Math.round(prob * 100)}%`;
          lit = prob;
        } else if (n.isInput) {
          text = "IN";
          lit = 0.4;
        } else if (stats) {
          if (stats.heatmaps && stats.heatmaps[pi]) {
            const hm = stats.heatmaps[pi];
            const peak = Math.max(...hm.values) / 255;
            text = peak.toFixed(2);
            lit = peak;
          } else if (stats.values && stats.values.length > pi) {
            const v = stats.values[pi];
            const max = Math.max(0.001, ...stats.values.map(Math.abs));
            text = (Math.abs(v) >= 10) ? v.toFixed(1) : v.toFixed(2);
            lit = Math.min(1, Math.abs(v) / max);
          }
        }
        mesh.material.emissiveIntensity = 0.12 + lit * 0.85;
        const c = mesh.userData.baseColor;
        const r = ((c >> 16) & 0xff) * lit;
        const g = ((c >> 8) & 0xff) * lit;
        const b = (c & 0xff) * lit;
        mesh.material.color.setRGB(r / 255, g / 255, b / 255);
        // grow slightly with score for visual emphasis
        const s = 1 + lit * 0.35;
        mesh.scale.set(s, s, s);

        if (labels[pi]) {
          setLabel(labels[pi], text, lit > 0.6 ? "#ffffff" : "#a0e9a0");
        }
      });
    });
  }

  window.NNV3D = { mount, updateScores, pause };
})();
