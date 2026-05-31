// Volumetric 3D visualization of the neural network.
//
// Each layer is rendered as a real 3D block whose dimensions reflect its
// tensor shape: a Conv2D(32,32,16) becomes a 3.2 x 3.2 x 1.6 box, a
// MaxPool that halves spatial dims shrinks accordingly, and a Dense(64)
// becomes a tall thin column. The input layer's front face is textured
// with the actual uploaded image so the student can literally see what
// the network is looking at.
//
// Floor grid + radial fog give the scene depth, three-point lighting
// (key + rim + ambient) shapes the blocks, and OrbitControls let the
// user drag-rotate / scroll-zoom to inspect from any angle.
//
// Public API:
//   window.NNV3D.mount(containerId, arch)
//   window.NNV3D.updateScores(nodes, predictionPayload, arch)
//   window.NNV3D.setInputImage(dataUri)   // optional — paints input face
//   window.NNV3D.pause()                  // stop render loop (e.g. on 2D switch)

(function () {
  if (typeof THREE === "undefined") {
    console.warn("[NNV3D] Three.js missing — 3D disabled");
    window.NNV3D = { mount: () => {}, updateScores: () => {}, pause: () => {}, setInputImage: () => {} };
    return;
  }

  const ACCENT = 0x28d600;
  const ACCENT_VEC = new THREE.Color(0x28d600);
  const COOL = 0x4ca8ff;
  const WHITE = 0xffffff;
  const POOL = 0xffb547;
  const DENSE = 0xff5fb3;

  // ---------- module state ----------
  let scene, camera, renderer, controls;
  let raf = null;
  let containerEl = null;
  let resizeObs = null;

  let archCache = null;
  let layerObjects = []; // [{ group, mesh, edges, label, shapeLabel, scoreSprite, kind, name, baseColor }]
  let connectionLines = [];
  let inputTextureMaterial = null;

  // ---------- helpers ----------
  function makeLabelTexture(text, color = "#ffffff", { font = 800, w = 512, h = 160 } = {}) {
    const cvs = document.createElement("canvas");
    cvs.width = w; cvs.height = h;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    ctx.font = `${font} 96px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // soft shadow for readability against any background
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 18;
    ctx.fillText(text, w / 2, h / 2);
    const tex = new THREE.CanvasTexture(cvs);
    tex.anisotropy = 8;
    return tex;
  }

  function makeChipTexture(text, color = "#28d600") {
    const cvs = document.createElement("canvas");
    cvs.width = 512; cvs.height = 180;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    // pill background
    ctx.fillStyle = "rgba(15, 18, 24, 0.92)";
    roundRect(ctx, 6, 14, cvs.width - 12, cvs.height - 28, 36);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    roundRect(ctx, 6, 14, cvs.width - 12, cvs.height - 28, 36);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = `bold 96px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cvs.width / 2, cvs.height / 2);
    const tex = new THREE.CanvasTexture(cvs);
    tex.anisotropy = 8;
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

  function makeSprite(tex, scale = [3.4, 1.1, 1]) {
    const mat = new THREE.SpriteMaterial({
      map: tex, transparent: true, depthTest: false, depthWrite: false,
    });
    const s = new THREE.Sprite(mat);
    s.scale.set(scale[0], scale[1], scale[2]);
    return s;
  }

  // map a tensor shape to a 3D box size {w,h,d}
  function shapeToBox(layer, kind) {
    const shape = (layer.output_shape || []).slice(1).filter((v) => v !== null && v !== "?" && typeof v !== "string");
    // 4D feature map: (H, W, C)
    if (shape.length === 3) {
      const [H, W, C] = shape;
      const scale = 0.10;
      // cap spatial so very large feature maps don't blow the scene
      const w = Math.max(0.6, Math.min(4.5, W * scale));
      const h = Math.max(0.6, Math.min(4.5, H * scale));
      const d = Math.max(0.4, Math.min(6.0, Math.log2(C + 2) * 0.75));
      return { w, h, d };
    }
    // 1D activations: dense / GAP output
    if (shape.length === 1) {
      const units = shape[0];
      // a tall thin column whose height grows with unit count
      const h = Math.max(0.7, Math.min(5.2, Math.log2(units + 2) * 0.9));
      return { w: 0.55, h, d: 0.55 };
    }
    return { w: 1, h: 1, d: 1 };
  }

  function colorFor(layer, kind, isOutput) {
    if (kind === "input") return WHITE;
    if (isOutput) return ACCENT;
    if (layer.type === "Conv2D") return ACCENT;
    if (layer.type === "MaxPooling2D" || layer.type === "GlobalAveragePooling2D") return POOL;
    if (layer.type === "Dense") return DENSE;
    return COOL;
  }

  function makeLayerBlock(layer, kind, isOutput, color) {
    const { w, h, d } = shapeToBox(layer, kind);

    const group = new THREE.Group();

    // Solid translucent body
    const geom = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x1e1e1e,
      transparent: true,
      opacity: 0.62,
      transmission: 0.4,
      thickness: 0.5,
      roughness: 0.35,
      metalness: 0.05,
      emissive: color,
      emissiveIntensity: 0.08,
      clearcoat: 0.6,
      clearcoatRoughness: 0.3,
    });
    const mesh = new THREE.Mesh(geom, mat);
    group.add(mesh);

    // Outlined wireframe edges so the prism reads at any angle
    const edgeGeom = new THREE.EdgesGeometry(geom);
    const edgeMat = new THREE.LineBasicMaterial({
      color: color, transparent: true, opacity: 0.85, linewidth: 2,
    });
    const edges = new THREE.LineSegments(edgeGeom, edgeMat);
    group.add(edges);

    // For Conv2D, draw a few inner "slice" planes to suggest channel stacking
    if (layer.type === "Conv2D" && d > 0.8) {
      const slices = Math.min(6, Math.max(2, Math.round(d * 2)));
      for (let i = 1; i < slices; i++) {
        const t = -d / 2 + (d * i) / slices;
        const sliceGeom = new THREE.PlaneGeometry(w * 0.92, h * 0.92);
        const sliceMat = new THREE.MeshBasicMaterial({
          color: color, transparent: true, opacity: 0.07, side: THREE.DoubleSide,
          depthWrite: false,
        });
        const slice = new THREE.Mesh(sliceGeom, sliceMat);
        slice.position.set(0, 0, t);
        group.add(slice);
      }
    }

    // Dense / GAP get horizontal "unit rings" for the same suggestion
    if ((layer.type === "Dense" || layer.type === "GlobalAveragePooling2D") && h > 1.2) {
      const rings = Math.min(8, Math.max(3, Math.round(h * 1.5)));
      for (let i = 1; i < rings; i++) {
        const t = -h / 2 + (h * i) / rings;
        const r = Math.max(w, d) * 0.55;
        const ringGeom = new THREE.RingGeometry(r * 0.85, r, 24);
        const ringMat = new THREE.MeshBasicMaterial({
          color: color, transparent: true, opacity: 0.18,
          side: THREE.DoubleSide, depthWrite: false,
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = t;
        group.add(ring);
      }
    }

    return { group, mesh, edges, w, h, d };
  }

  function clearScene() {
    if (!scene) return;
    layerObjects.forEach((lo) => {
      if (lo.group) scene.remove(lo.group);
      if (lo.label) scene.remove(lo.label);
      if (lo.shapeLabel) scene.remove(lo.shapeLabel);
      if (lo.scoreSprite) scene.remove(lo.scoreSprite);
    });
    connectionLines.forEach((l) => scene.remove(l));
    layerObjects = [];
    connectionLines = [];
  }

  function buildScene(arch) {
    archCache = arch;
    clearScene();

    const layers = arch.layers;
    const N = layers.length;
    // First pass: compute every layer's box so we can place them with a gap.
    const boxes = layers.map((layer, i) => {
      const isOutput = i === N - 1;
      const kind = i === 0 ? "input" : (isOutput ? "output" : "hidden");
      const color = colorFor(layer, kind, isOutput);
      return { layer, isOutput, isInput: i === 0, kind, color, box: shapeToBox(layer, kind) };
    });

    // X positions: cumulative with a fixed gap; centre the whole thing afterwards.
    const gap = 1.5;
    let xCursor = 0;
    const xPositions = boxes.map((b) => {
      const x = xCursor + b.box.w / 2;
      xCursor += b.box.w + gap;
      return x;
    });
    const totalW = xCursor - gap;
    const offsetX = -totalW / 2;

    boxes.forEach((b, i) => {
      const x = xPositions[i] + offsetX;
      const lb = makeLayerBlock(b.layer, b.kind, b.isOutput, b.color);
      lb.group.position.set(x, 0, 0);
      scene.add(lb.group);

      // Layer name label above
      const nameTex = makeLabelTexture(
        b.layer.type.replace("2D", "").toUpperCase(),
        b.kind === "input" ? "#ffffff" : "#" + b.color.toString(16).padStart(6, "0")
      );
      const nameSpr = makeSprite(nameTex, [Math.max(2.2, lb.w + 0.4), 0.55, 1]);
      nameSpr.position.set(x, lb.h / 2 + 0.7, 0);
      scene.add(nameSpr);

      // Shape label just below the name
      const shape = (b.layer.output_shape || []).slice(1)
        .filter((v) => v !== null && v !== "?" && typeof v !== "string")
        .join("×");
      const shapeTex = makeLabelTexture(shape, "rgba(255,255,255,0.7)", { font: 600 });
      const shapeSpr = makeSprite(shapeTex, [Math.max(2.2, lb.w + 0.4), 0.42, 1]);
      shapeSpr.position.set(x, lb.h / 2 + 0.15, 0);
      scene.add(shapeSpr);

      // Score chip — below the block, populated after each prediction
      const initial = b.isInput ? "IN" : (b.isOutput ? b.layer.name : "—");
      const initialColor = b.isInput ? "#ffffff" : "#" + b.color.toString(16).padStart(6, "0");
      const chipTex = makeChipTexture(initial, initialColor);
      const chipSpr = makeSprite(chipTex, [2.0, 0.65, 1]);
      chipSpr.position.set(x, -lb.h / 2 - 0.65, 0);
      scene.add(chipSpr);

      // For the OUTPUT layer, also place a small class-name billboard at the
      // top-right of the block so the user immediately understands what each
      // probability refers to.
      let classLabel = null;
      if (b.isOutput && arch.classes && arch.classes.length) {
        // pick the class name for this block when output is shown as a single block;
        // but here output is one block (Dense 6) — we'll show a row of per-class chips.
      }

      layerObjects.push({
        group: lb.group,
        mesh: lb.mesh,
        edges: lb.edges,
        w: lb.w, h: lb.h, d: lb.d,
        label: nameSpr,
        shapeLabel: shapeSpr,
        scoreSprite: chipSpr,
        kind: b.kind,
        isInput: b.isInput,
        isOutput: b.isOutput,
        name: b.layer.name,
        layer: b.layer,
        baseColor: b.color,
        position: new THREE.Vector3(x, 0, 0),
      });
    });

    // Connecting lines between adjacent layer centres
    const lineMat = new THREE.LineBasicMaterial({
      color: ACCENT, transparent: true, opacity: 0.35,
    });
    for (let i = 0; i < layerObjects.length - 1; i++) {
      const a = layerObjects[i];
      const bnext = layerObjects[i + 1];
      const pa = new THREE.Vector3(a.position.x + a.w / 2, 0, 0);
      const pb = new THREE.Vector3(bnext.position.x - bnext.w / 2, 0, 0);
      const geom = new THREE.BufferGeometry().setFromPoints([pa, pb]);
      const line = new THREE.Line(geom, lineMat);
      scene.add(line);
      connectionLines.push(line);
    }

    // Per-class chip ribbon next to the output block
    if (arch.classes && arch.classes.length) {
      const last = layerObjects[layerObjects.length - 1];
      last.classChips = [];
      const ribbonX = last.position.x + last.w / 2 + 1.4;
      const rowSpan = Math.min(5, arch.classes.length * 0.7);
      const rowStep = rowSpan / Math.max(1, arch.classes.length - 1);
      arch.classes.forEach((cls, k) => {
        const y = rowSpan / 2 - k * rowStep;
        const tex = makeChipTexture(cls, "#28d600");
        const spr = makeSprite(tex, [1.8, 0.6, 1]);
        spr.position.set(ribbonX, y, 0);
        scene.add(spr);
        // probability chip beside it
        const pTex = makeChipTexture("—", "#ffffff");
        const pSpr = makeSprite(pTex, [1.3, 0.6, 1]);
        pSpr.position.set(ribbonX + 1.9, y, 0);
        scene.add(pSpr);
        last.classChips.push({ name: cls, probSprite: pSpr });
      });
    }

    // refit camera to the new scene
    fitCameraToScene();
  }

  function fitCameraToScene() {
    if (!camera || layerObjects.length === 0) return;
    const totalW = layerObjects[layerObjects.length - 1].position.x + 4 - layerObjects[0].position.x;
    const maxH = Math.max(...layerObjects.map((o) => o.h));
    const dist = Math.max(totalW * 0.9, maxH * 2.2, 14);
    camera.position.set(0, maxH * 0.6 + 1.2, dist);
    camera.lookAt(0, 0, 0);
    if (controls) controls.target.set(0, 0, 0);
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

  function makeFloor() {
    const grid = new THREE.GridHelper(40, 40, 0x28d600, 0x223322);
    grid.position.y = -5;
    grid.material.opacity = 0.18;
    grid.material.transparent = true;
    scene.add(grid);
    // soft glow disk in the centre of the grid
    const ringGeom = new THREE.RingGeometry(0.5, 8, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: ACCENT, transparent: true, opacity: 0.08,
      side: THREE.DoubleSide, depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -4.99;
    scene.add(ring);
  }

  function mount(containerId, arch) {
    containerEl = document.getElementById(containerId);
    if (!containerEl || !arch) return;

    if (!renderer) {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x06090d);
      scene.fog = new THREE.FogExp2(0x06090d, 0.025);

      camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace || renderer.outputColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.1;
      containerEl.appendChild(renderer.domElement);

      // Three-point lighting
      scene.add(new THREE.AmbientLight(0xffffff, 0.35));
      const key = new THREE.DirectionalLight(0xffffff, 0.8);
      key.position.set(8, 12, 10);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0x28d600, 0.55);
      rim.position.set(-10, 4, -8);
      scene.add(rim);
      const fill = new THREE.PointLight(0x6fffe9, 0.55, 40);
      fill.position.set(0, 4, 12);
      scene.add(fill);

      if (THREE.OrbitControls) {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.rotateSpeed = 0.8;
        controls.zoomSpeed = 0.7;
        controls.minDistance = 6;
        controls.maxDistance = 60;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.45;
        controls.target.set(0, 0, 0);
        // stop auto-rotate as soon as user interacts
        renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; });
      }

      makeFloor();
      resizeObs = new ResizeObserver(ensureRendererSized);
      resizeObs.observe(containerEl);
    }

    buildScene(arch);
    ensureRendererSized();
    if (inputTextureMaterial && layerObjects.length) applyInputTextureToFirst();
    if (!raf) loop();
  }

  function applyInputTextureToFirst() {
    if (!layerObjects.length || !inputTextureMaterial) return;
    const first = layerObjects[0];
    if (!first || !first.mesh) return;
    // BoxGeometry's per-face material: replace front face (idx 4) with image,
    // keep the rest of the body translucent.
    const bodyMat = first.mesh.material;
    const faceMats = [bodyMat, bodyMat, bodyMat, bodyMat, inputTextureMaterial, bodyMat];
    first.mesh.material = faceMats;
    first.mesh.material.needsUpdate = true;
  }

  function setInputImage(dataUri) {
    if (!dataUri) return;
    const loader = new THREE.TextureLoader();
    loader.load(dataUri, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace || tex.colorSpace;
      tex.anisotropy = 8;
      inputTextureMaterial = new THREE.MeshBasicMaterial({ map: tex });
      if (layerObjects.length) applyInputTextureToFirst();
    });
  }

  function updateScores(nodes, data, arch) {
    if (!scene || !layerObjects.length) return;
    archCache = arch || archCache;
    const layerByName = {};
    (data.layers || []).forEach((l) => (layerByName[l.name] = l));

    layerObjects.forEach((lo) => {
      // determine per-block "score" + label
      let chipText = "—";
      let chipColor = "#" + lo.baseColor.toString(16).padStart(6, "0");
      let lit = 0;

      if (lo.isOutput) {
        // For the output block, take the top class probability as the headline
        const top = (data.predictions || [])[0];
        if (top) {
          chipText = `${top.label} · ${Math.round(top.prob * 100)}%`;
          lit = top.prob;
          chipColor = "#28d600";
        }
        // also fill the per-class probability chips beside the block
        if (lo.classChips) {
          lo.classChips.forEach(({ name, probSprite }) => {
            const e = (data.predictions || []).find((p) => p.label === name);
            const prob = e ? e.prob : 0;
            const pct = `${Math.round(prob * 100)}%`;
            probSprite.material.map = makeChipTexture(pct, prob > 0.5 ? "#28d600" : "#ffffff");
            probSprite.material.needsUpdate = true;
            const s = 1 + prob * 0.6;
            probSprite.scale.set(1.3 * s, 0.6 * s, 1);
          });
        }
      } else if (lo.isInput) {
        chipText = `${arch && arch.input_size ? arch.input_size + "×" + arch.input_size : "input"}`;
        lit = 0.5;
        chipColor = "#ffffff";
      } else {
        const stats = layerByName[lo.name];
        if (stats) {
          // headline score = max activation in the layer
          const m = stats.max != null ? stats.max : 0;
          const meanV = stats.mean != null ? stats.mean : 0;
          chipText = `max ${m.toFixed(2)}  ·  μ ${meanV.toFixed(2)}`;
          lit = Math.max(0, Math.min(1, m / 6)); // soft normalize for visual scaling
        }
      }

      // refresh chip texture
      if (lo.scoreSprite) {
        lo.scoreSprite.material.map = makeChipTexture(chipText, chipColor);
        lo.scoreSprite.material.needsUpdate = true;
      }
      // tint block: emissive ramps with activation
      if (lo.mesh) {
        const mat = Array.isArray(lo.mesh.material) ? lo.mesh.material[0] : lo.mesh.material;
        if (mat && mat.emissive) {
          mat.emissiveIntensity = 0.08 + lit * 0.85;
        }
      }
      if (lo.edges && lo.edges.material) {
        lo.edges.material.opacity = 0.55 + lit * 0.45;
      }
      // subtle scale pulse with activation
      const s = 1 + lit * 0.08;
      lo.group.scale.set(s, s, s);
    });
  }

  window.NNV3D = { mount, updateScores, pause, setInputImage };
})();
