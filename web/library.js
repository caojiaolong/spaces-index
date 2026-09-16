// Library presentation. All illustrations are generated locally from metadata.
// This file is loaded before app.js; functions run after the catalogue is ready.

const SERIES_ARTWORKS = [
  ["matrix", "MATRIX STUDIES"], ["diffusion", "PARTICLE FIELD"],
  ["geometry", "SPATIAL GEOMETRY"], ["curve", "CONTINUOUS FORMS"],
  ["contours", "CONTOUR ATLAS"], ["spiral", "GROWTH & ORDER"],
  ["constellation", "CONNECTED IDEAS"], ["lissajous", "HARMONIC MOTION"],
  ["waves", "WAVE INTERFERENCE"], ["facets", "PLANAR STRUCTURES"],
  ["orbits", "ORBITAL SYSTEMS"], ["lattice", "LATTICE STUDIES"],
  ["spectrum", "FREQUENCY SPECTRUM"], ["rosette", "RADIAL SYMMETRY"],
];

function seriesSeed(value) {
  let hash = 2166136261;
  for (const char of value) hash = Math.imul(hash ^ char.codePointAt(0), 16777619);
  // Mix the lower bits too, so similar series identifiers spread across forms.
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  return (hash ^ (hash >>> 16)) >>> 0;
}

function seriesAppearance(series) {
  // Independent seeds keep artwork and colour varied, but stable across routes,
  // reloads and catalogue updates. No random state or storage is needed.
  const identity = String(series.id || series.name || "series");
  const index = seriesSeed(`art:${identity}`) % SERIES_ARTWORKS.length;
  const [kind, label] = SERIES_ARTWORKS[index];
  return {
    kind, label, number: String(index + 1).padStart(2, "0"),
    hue: seriesSeed(`colour:${identity}`) % 360,
    saturation: 28 + seriesSeed(`tone:${identity}`) % 15,
    seed: seriesSeed(`shape:${identity}`),
  };
}

function makeSeriesArt(series, appearance = seriesAppearance(series)) {
  const { kind, seed } = appearance;
  const svg = createSvgElement("svg", { viewBox: "0 0 360 190", "aria-hidden": "true", focusable: "false" });
  let state = seed || 1;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
  const phase = random() * Math.PI * 2;
  const path = (d, opacity = .55, width = 1, fill = "none") => svg.append(createSvgElement("path", {
    d, fill, stroke: "currentColor", "stroke-width": width, "stroke-linecap": "round", "stroke-linejoin": "round", opacity,
  }));
  const dot = (cx, cy, r = 2, opacity = .65) => svg.append(createSvgElement("circle", { cx, cy, r, fill: "currentColor", opacity }));
  const plot = (point, steps = 160, closed = false) => Array.from({ length: steps + 1 }, (_, i) => {
    const [x, y] = point(i / steps);
    return `${i ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ") + (closed ? "Z" : "");
  if (kind === "diffusion") {
    for (let x = 0; x < 17; x++) {
      for (let y = 0; y < 8; y++) {
        const spread = 1 - x / 19;
        const noise = Math.sin(x * 73 + y * 31 + seed);
        svg.append(createSvgElement("circle", {
          cx: 24 + x * 19 + noise * 9 * spread,
          cy: 28 + y * 19 + Math.cos(x * 29 + y * 71 + seed) * 12 * spread,
          r: 1.1 + x / 13, fill: "currentColor", opacity: .2 + x / 25,
        }));
      }
    }
  } else if (kind === "matrix") {
    for (let x = 0; x < 12; x++) {
      for (let y = 0; y < 6; y++) {
        svg.append(createSvgElement("rect", {
          x: 58 + x * 21, y: 25 + y * 24, width: 16, height: 19, rx: 1,
          fill: "currentColor", opacity: .12 + random() * .65,
        }));
      }
    }
    svg.append(createSvgElement("path", { d: "M41 25H32V164H41 M321 25H330V164H321", fill: "none", stroke: "currentColor", "stroke-width": 1.5 }));
  } else if (kind === "geometry") {
    for (let i = 0; i < 12; i++) {
      svg.append(createSvgElement("ellipse", {
        cx: 180, cy: 95, rx: 118, ry: 12 + i * 5,
        transform: `rotate(${i * 4 - 22 + seed % 9} 180 95)`, fill: "none", stroke: "currentColor", opacity: .2 + i / 24,
      }));
    }
  } else if (kind === "curve") {
    for (let i = 0; i < 6; i++) {
      svg.append(createSvgElement("path", {
        d: `M20 ${30 + i * 16} C85 ${-40 + i * 24 + seed % 20} 100 ${210 - i * 11} 165 ${135 - i * 9} S240 ${38 + i * 14} 340 ${82 + i * 10 + seed % 15}`,
        fill: "none", stroke: "currentColor", "stroke-width": i === 2 ? 2.5 : 1, opacity: .2 + i / 9,
      }));
    }
  } else if (kind === "contours") {
    const lobes = 3 + seed % 3;
    for (let ring = 1; ring <= 11; ring++) {
      path(plot((t) => {
        const angle = t * Math.PI * 2;
        const r = ring / 11 * (1 + .13 * Math.sin(lobes * angle + phase) + .06 * Math.cos(7 * angle));
        return [180 + Math.cos(angle) * 120 * r, 95 + Math.sin(angle) * 65 * r];
      }, 120, true), .18 + ring * .05, ring % 4 === 0 ? 1.5 : .8);
    }
  } else if (kind === "spiral") {
    const angleStep = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < 180; i++) {
      const radius = Math.sqrt(i / 180);
      const angle = i * angleStep + phase;
      dot(180 + Math.cos(angle) * 123 * radius, 95 + Math.sin(angle) * 77 * radius, 1.1 + radius * 1.8, .25 + radius * .5);
    }
  } else if (kind === "constellation") {
    const points = Array.from({ length: 24 }, (_, i) => [
      35 + (i % 6) * 55 + random() * 16, 25 + Math.floor(i / 6) * 42 + random() * 16,
    ]);
    points.forEach(([x, y], i) => {
      points.slice(i + 1).forEach(([nx, ny]) => {
        const distance = Math.hypot(nx - x, ny - y);
        if (distance < 83) path(`M${x} ${y}L${nx} ${ny}`, .15 + (1 - distance / 83) * .4);
      });
      dot(x, y, 1.7 + random() * 2.4, .8);
    });
  } else if (kind === "lissajous") {
    const frequency = [3, 5, 7][seed % 3];
    for (let i = 0; i < 4; i++) {
      path(plot((t) => [180 + 129 * Math.sin(t * Math.PI * 2 * frequency + phase + i * .13),
        95 + 73 * Math.sin(t * Math.PI * 4)], 320, true), .25 + i * .12, .85);
    }
  } else if (kind === "waves") {
    for (let i = 0; i < 17; i++) {
      path(plot((t) => [22 + t * 316, 36 + i * 7.3 + Math.sin(t * Math.PI * 3 + phase + i * .3) * 15
        + Math.cos(t * Math.PI * 6 - i * .2) * 7], 90), .2 + (i % 6) * .1, i % 5 === 0 ? 1.7 : .8);
    }
  } else if (kind === "facets") {
    const rows = Array.from({ length: 4 }, (_, y) => Array.from({ length: 6 }, (_, x) => [
      32 + x * 58 + (x > 0 && x < 5 ? random() * 22 - 11 : 0),
      22 + y * 48 + (y > 0 && y < 3 ? random() * 22 - 11 : 0),
    ]));
    for (let y = 0; y < 3; y++) for (let x = 0; x < 5; x++) {
      const [a, b, c, d] = [rows[y][x], rows[y][x + 1], rows[y + 1][x], rows[y + 1][x + 1]];
      for (const triangle of [[a, b, c], [b, d, c]]) {
        path(`M${triangle.map((p) => p.join(" ")).join("L")}Z`, .08 + random() * .43, .6, "currentColor");
      }
    }
  } else if (kind === "orbits") {
    for (let i = 0; i < 8; i++) {
      const rx = 26 + i * 15;
      const ry = 13 + i * 8;
      const angle = phase + i * 1.9;
      svg.append(createSvgElement("ellipse", { cx: 180, cy: 95, rx, ry, fill: "none", stroke: "currentColor", opacity: .2 + i * .05 }));
      dot(180 + Math.cos(angle) * rx, 95 + Math.sin(angle) * ry, 2.5 + random() * 2.5, .8);
    }
    dot(180, 95, 7, .75);
  } else if (kind === "lattice") {
    const tilt = .3 + random() * .25;
    const point = (x, y) => [180 + (x - y) * 22, 95 + (x + y - 6) * 11 + Math.sin(x * tilt + phase) * 11];
    for (let x = 0; x < 7; x++) for (let y = 0; y < 7; y++) {
      const p = point(x, y);
      if (x < 6) path(`M${p.join(" ")}L${point(x + 1, y).join(" ")}`, .4);
      if (y < 6) path(`M${p.join(" ")}L${point(x, y + 1).join(" ")}`, .4);
      dot(...p, (x + y) % 3 === 0 ? 2.7 : 1.5, .7);
    }
  } else if (kind === "spectrum") {
    path("M22 95H338", .2);
    for (let i = 0; i < 45; i++) {
      const envelope = Math.sin((i + 1) / 46 * Math.PI);
      const height = 8 + envelope * (12 + random() * 53);
      const x = 26 + i * 7;
      path(`M${x} ${95 - height}V${95 + height}`, .3 + random() * .5, 2.4);
    }
  } else if (kind === "rosette") {
    const petals = 4 + seed % 5;
    for (let i = 0; i < 7; i++) {
      path(plot((t) => {
        const angle = t * Math.PI * 2;
        const radius = (42 + i * 8) * (.68 + .32 * Math.cos(petals * angle + phase));
        return [180 + radius * 1.5 * Math.cos(angle), 95 + radius * .85 * Math.sin(angle)];
      }, 240, true), .2 + i * .08, i === 6 ? 1.6 : .8);
    }
  }
  return svg;
}

function makeSeriesCover(series) {
  const appearance = seriesAppearance(series);
  const cover = createElement("div", { className: `series-cover cover-${appearance.kind}`, attrs: { "aria-hidden": "true" } }, [
    createElement("div", { className: "cover-caption" }, [
      createElement("span", { text: appearance.label }), createElement("span", { text: appearance.number }),
    ]),
    createElement("div", { className: "cover-title", text: series.name }),
    makeSeriesArt(series, appearance),
    createElement("div", { className: "cover-baseline" }, [
      createElement("span", { text: "科学空间 · 系列文集" }), createElement("span", { text: `${series.count} 篇` }),
    ]),
  ]);
  cover.style.setProperty("--cover-hue", appearance.hue);
  cover.style.setProperty("--cover-saturation", `${appearance.saturation}%`);
  return cover;
}

function seriesReadingState(catalog, series, readIds = ui.readPostIds) {
  const posts = postsForSeries(catalog, series);
  return {
    total: posts.length,
    read: posts.filter((post) => readIds.has(String(post.id))).length,
    next: posts.find((post) => !readIds.has(String(post.id))) || null,
  };
}

function seriesChapterHref(series, post) {
  return hrefFor(`/series/${encodeURIComponent(series.id)}`, new URLSearchParams(post ? { chapter: post.id } : {}));
}

function updateSeriesProgress(root, series) {
  const state = seriesReadingState(ui.catalog, series);
  const label = root.querySelector(".progress-label");
  const bar = root.querySelector("progress");
  const link = root.querySelector(".continue-link");
  label.textContent = `已读 ${state.read} / ${state.total}`;
  bar.max = Math.max(1, state.total);
  bar.value = state.read;
  bar.setAttribute("aria-label", `${series.name}：已读 ${state.read} 篇，共 ${state.total} 篇`);
  const text = !state.next ? "重温系列 →" : getPostProgress(state.next.id) > 0 ? "继续阅读 →" : state.read ? "继续下一篇 →" : "从第一篇开始 →";
  link.replaceWith(state.next
    ? makePostLink(state.next, null, { className: "continue-link", text, label: `${text} ${state.next.title}` })
    : createElement("a", { className: "continue-link", href: seriesChapterHref(series), text, attrs: { "aria-label": `重温 ${series.name}` } }));
}

function makeSeriesProgress(series) {
  const root = createElement("div", { className: "series-progress", dataset: { seriesProgress: series.id } }, [
    createElement("div", { className: "progress-meta" }, [
      createElement("span", { className: "progress-label" }),
      createElement("a", { className: "continue-link" }),
    ]),
    createElement("progress", { attrs: { max: 1, value: 0 } }),
  ]);
  updateSeriesProgress(root, series);
  return root;
}

function makeSeriesCard(series) {
  return createElement("article", { className: "series-card" }, [
    createElement("a", { className: "series-cover-link", href: seriesHref(series.id), attrs: { tabindex: "-1", "aria-hidden": "true" } }, makeSeriesCover(series)),
    createElement("div", { className: "series-card-body" }, [
      createElement("span", { className: "card-kicker", text: series.topic || "专题系列" }),
      createElement("h3", {}, createElement("a", { href: seriesHref(series.id), text: series.name })),
      createElement("div", { className: "series-meta" }, [
        createElement("span", { text: `${series.count} 篇文章` }),
        series.startDate && series.endDate ? createElement("span", { text: `${series.startDate.slice(0, 4)} — ${series.endDate.slice(0, 4)}` }) : null,
      ]),
      makeSeriesProgress(series),
    ]),
  ]);
}

function fillContinueReading(root, catalog) {
  const started = catalog.series.map((series) => ({ series, state: seriesReadingState(catalog, series) }))
    .filter(({ series, state }) => state.next && (state.read > 0 || series.postIds.some(id => getPostProgress(id) > 0)))
    .sort((a, b) => Number(b.series.postIds.includes(ui.lastReadPostId)) - Number(a.series.postIds.includes(ui.lastReadPostId)) || dateNumber(b.series.endDate) - dateNumber(a.series.endDate));
  root.hidden = !started.length;
  if (!started.length) { root.replaceChildren(); return; }
  const { series, state } = started[0];
  root.replaceChildren(
    createElement("span", { className: "continue-caption", text: "接着上次的思考" }),
    makePostLink(state.next),
    makePostLink(state.next, null, { className: "continue-count", text: `${state.read} / ${state.total} 篇 · 继续阅读 →`, label: `继续阅读：${state.next.title}` }),
  );
}

function makeTopicCard(topic) {
  return createElement("a", { className: "topic-card", href: topicHref(topic.name) }, [
    createElement("h3", { text: topic.name }),
    createElement("span", { text: `${topic.count} 篇 ↗` }),
  ]);
}

function topicConnections(catalog, name) {
  const source = catalog.posts.filter((post) => post.topics.includes(name));
  const sourceIds = new Set(source.map((post) => post.id));
  const tags = new Map(source.flatMap((post) => post.sourceTags).map((tag) => [normalizeText(tag), tag]).filter(([key]) => key));
  const tagCounts = (posts) => {
    const counts = new Map();
    posts.forEach((post) => new Set(post.sourceTags.map(normalizeText).filter(Boolean)).forEach((tag) => counts.set(tag, (counts.get(tag) || 0) + 1)));
    return counts;
  };
  const sourceCounts = tagCounts(source);
  return catalog.topics.filter((topic) => topic.name !== name && topic.count).map((topic) => {
    const posts = catalog.posts.filter((post) => post.topics.includes(topic.name));
    const targetTags = new Set(posts.flatMap((post) => post.sourceTags).map(normalizeText));
    const targetCounts = tagCounts(posts);
    return {
      topic,
      postCount: new Set(posts.filter((post) => sourceIds.has(post.id)).map((post) => post.id)).size,
      tags: [...tags].filter(([key]) => targetTags.has(key))
        .sort(([a], [b]) => Math.min(sourceCounts.get(b), targetCounts.get(b)) - Math.min(sourceCounts.get(a), targetCounts.get(a)) || a.localeCompare(b, "zh-CN"))
        .map(([, label]) => label),
    };
  }).filter((item) => item.postCount || item.tags.length)
    .sort((a, b) => b.postCount - a.postCount || b.tags.length - a.tags.length || a.topic.name.localeCompare(b.topic.name, "zh-CN"));
}

function makeTopicConnections(catalog, initialName, { interactive = false } = {}) {
  const section = createElement("section", { className: "connections-section" });
  const heading = createSectionHeading("Across disciplines", "思考的交汇处", "共同文章表示被同时归入两个主题；共同标签表示两个主题都使用过该标签。它们不代表知识的先修关系。");
  const panel = createElement("div", { className: "connections-panel" });
  function draw(name) {
    const connections = topicConnections(catalog, name).slice(0, 3);
    const links = createElement("div", { className: "connection-list" });
    connections.forEach(({ topic, postCount, tags }) => {
      const params = new URLSearchParams({ topic: name, with: topic.name });
      const row = createElement("article", { className: `connection-row${postCount ? "" : " tags-only"}` }, [
        createElement("a", { className: "connection-title", href: topicHref(topic.name), text: `${topic.name} ↗` }),
        createElement("div", { className: "connection-evidence" }, [
          postCount ? createElement("a", { href: hrefFor("/explore", params), text: `共同收录 ${postCount} 篇 →` }) : createElement("span", { text: "通过共同标签相连" }),
          createElement("span", { text: `${tags.length} 个共同标签` }),
        ]),
      ]);
      if (tags.length) row.append(createElement("div", { className: "connection-tags" }, tags.slice(0, 3).map((tag) => {
        const query = new URLSearchParams({ tag });
        query.append("topic", name); query.append("topic", topic.name);
        return createElement("a", { href: hrefFor("/explore", query), text: `# ${tag}`, attrs: { "aria-label": `浏览 ${name} 与 ${topic.name} 中带有标签 ${tag} 的文章` } });
      })));
      links.append(row);
    });
    panel.replaceChildren(createElement("div", { className: "connection-origin" }, [
      createElement("span", { className: "card-kicker", text: "从这里出发" }),
      createElement("a", { href: topicHref(name), text: name }),
      createElement("span", { className: "connection-legend", text: "实线 · 共同文章\n虚线 · 仅共同标签" }),
    ]), connections.length ? links : createElement("p", { className: "page-description", text: "当前元数据中还没有发现与其他主题的共同文章或标签。" }));
  }
  if (interactive) heading.append(createSelect({
    id: "connection-topic", label: "选择起点主题", value: initialName,
    options: catalog.topics.filter((topic) => topic.count).map((topic) => ({ value: topic.name, label: topic.name })),
    onChange: (name) => {
      ui.focusAfterRender = null;
      const updated = makeTopicConnections(catalog, name, { interactive: true });
      section.replaceWith(updated);
      updated.querySelector("button")?.focus({ preventScroll: true });
    },
  }));
  draw(initialName);
  section.append(heading, panel);
  return section;
}

function makeSculpture() {
  // Keep one SVG mesh; vary its surface and projection without replacing nodes.
  const svg = createSvgElement("svg", { viewBox: "0 0 520 370", class: "atlas-sculpture", "aria-hidden": "true", focusable: "false" });
  const mesh = createSvgElement("g", { class: "sculpture-mesh", fill: "none", "stroke-linecap": "round" });
  const curves = [];
  const sample = (u, v) => ({ cu: Math.cos(u), su: Math.sin(u), s2u: Math.sin(2 * u), c3u: Math.cos(3 * u), cv: Math.cos(v), sv: Math.sin(v) });
  function addCurve(points, attrs) {
    const path = createSvgElement("path", attrs);
    curves.push({ path, points }); mesh.append(path);
  }
  for (let line = 0; line < 26; line++) {
    const v = line / 26 * Math.PI * 2;
    addCurve(Array.from({ length: 113 }, (_, step) => sample(step / 112 * Math.PI * 2, v)),
      { class: line === 4 ? "sculpture-highlight" : "sculpture-latitude", opacity: .28 + (Math.sin(v) + 1) * .22 });
  }
  for (let line = 0; line < 48; line++) {
    const u = line / 48 * Math.PI * 2;
    addCurve(Array.from({ length: 41 }, (_, step) => sample(u, step / 40 * Math.PI * 2)), { class: "sculpture-longitude" });
  }
  const shadow = createSvgElement("ellipse", { cx: 264, cy: 323, rx: 115, ry: 13, class: "sculpture-shadow" });
  svg.append(
    shadow,
    createSvgElement("path", { d: "M32 34H50 M41 25V43 M470 318H488 M479 309V327", class: "sculpture-registration" }),
    mesh,
  );
  svg.drawSurface = seconds => {
    const phase = seconds * Math.PI * 2 / 42;
    const breath = Math.sin(phase);
    const stretch = 1 + .12 * Math.sin(phase * .7);
    const tilt = .98 + .18 * Math.sin(phase * .8);
    const roll = .43 + .08 * Math.sin(phase * .6);
    const spin = seconds * .075;
    const ct = Math.cos(tilt), st = Math.sin(tilt), cr = Math.cos(roll), sr = Math.sin(roll);
    const cs = Math.cos(spin), ss = Math.sin(spin);
    const scale = 108 / Math.max(1, stretch);
    let extentX = 0, extentY = 0;
    const projected = curves.map(({ path, points }) => ({ path,
      points: points.map(p => {
        const tube = .53 + .10 * breath + .06 * breath * p.c3u;
        const radius = 1.46 - .10 * breath + tube * p.cv;
        const x = radius * p.cu * stretch, y = radius * p.su / stretch;
        const z = tube * p.sv + (.22 + .18 * Math.sin(phase * .9)) * p.s2u;
        const rotatedX = x * cs - y * ss, rotatedY = x * ss + y * cs;
        const tiltedY = rotatedY * ct - z * st;
        const perspective = 1 + (rotatedY * st + z * ct) * .09;
        const px = (rotatedX * cr - tiltedY * sr) * scale * perspective;
        const py = (rotatedX * sr + tiltedY * cr) * scale * perspective;
        extentX = Math.max(extentX, Math.abs(px)); extentY = Math.max(extentY, Math.abs(py));
        return [px, py];
      }),
    }));
    // Fit every pose using model coordinates, without measuring layout per frame.
    const fit = Math.min(1, 236 / extentX, 160 / extentY);
    for (const { path, points } of projected) {
      path.setAttribute("d", points.map(([x, y], index) =>
        `${index ? "L" : "M"}${(260 + x * fit).toFixed(1)},${(183 + y * fit).toFixed(1)}`
      ).join(" "));
    }
    shadow.setAttribute("rx", (115 + breath * 9).toFixed(1));
  };
  svg.drawSurface(0);
  return svg;
}

function mountHeroMotion(view) {
  const svg = view.querySelector(".atlas-sculpture");
  if (!svg || !window.IntersectionObserver) return () => {};
  const controller = new AbortController();
  const { signal } = controller;
  let frame = 0, previousTime = null, elapsed = 0, visible = false;
  const canAnimate = () => visible && !document.hidden && !reduceMotion.matches && !signal.aborted;
  function tick(now) {
    frame = 0;
    if (!view.isConnected || !canAnimate()) { previousTime = null; return; }
    if (previousTime === null) previousTime = now;
    // 24 fps is enough for a slow surface; pause time never advances its shape.
    if (now - previousTime >= 1000 / 24) {
      elapsed += Math.min(now - previousTime, 100) / 1000;
      previousTime = now;
      svg.drawSurface(elapsed);
    }
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    if (canAnimate()) {
      if (!frame) frame = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(frame); frame = 0; previousTime = null;
    }
  }
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting; sync();
  }, { threshold: 0 });
  observer.observe(svg);
  document.addEventListener("visibilitychange", sync, { signal });
  reduceMotion.addEventListener("change", sync, { signal });
  sync();
  return () => { controller.abort(); cancelAnimationFrame(frame); observer.disconnect(); };
}

function makeHeroVisual(catalog) {
  const figure = createElement("div", { className: "knowledge-sketch" });
  figure.append(createElement("div", { className: "sketch-caption" }, [
    createElement("span", { text: "THE SHAPE OF IDEAS" }),
    createElement("span", { text: "FIG. 01" }),
  ]), makeSculpture());
  const names = ["大模型与Transformer", "生成模型", "数学工具"].filter((name) => catalog.topics.some((topic) => topic.name === name && topic.count));
  figure.append(createElement("div", { className: "sketch-links" }, names.map((name, index) => createElement("a", {
    href: topicHref(name),
  }, [
    createElement("span", { className: "sketch-link-index", text: String(index + 1).padStart(2, "0") }),
    createElement("span", { text: name === "大模型与Transformer" ? "Transformer" : name }),
    createElement("span", { text: "↗", attrs: { "aria-hidden": "true" } }),
  ]))));
  return figure;
}

function renderHome(catalog) {
  const view = createElement("div", { className: "view home-view" });
  const hero = createElement("section", { className: "hero page-shell" });
  hero.append(createElement("div", { className: "hero-masthead" }, [
    createElement("span", { text: "数学 · 人工智能 · 自然科学" }),
    createElement("span", { className: "live-caption", text: "持续生长的知识索引" }),
  ]));
  const search = createElement("input", { id: "home-search", type: "search", placeholder: "搜索标题、系列、标签或小结", attrs: { "aria-label": "搜索文章", autocomplete: "off" } });
  const form = createElement("form", { className: "home-search", attrs: { role: "search" }, on: { submit: (event) => {
    event.preventDefault();
    setHash("/explore", new URLSearchParams(search.value.trim() ? { q: search.value.trim() } : {}));
  } } }, [
    createElement("span", { text: "⌕", attrs: { "aria-hidden": "true" } }), search,
    createElement("button", { type: "submit", text: "搜索 →", attrs: { "aria-label": "搜索文章" } }),
  ]);
  const copy = createElement("div", { className: "hero-copy-block" }, [
    createElement("p", { className: "eyebrow hero-eyebrow", text: "A living atlas of knowledge" }),
    createElement("h1", {}, [createElement("span", { text: "循着问题，" }), createElement("span", { text: "发现科学空间。" })]),
    createElement("p", { className: "hero-copy", text: "从一篇文章，走进一个思想世界。沿着苏剑林的主题与系列，发现值得深入的下一步。" }),
    form,
    createElement("div", { className: "home-quick-links" }, [
      createElement("span", { text: "从这里开始" }),
      ...["Transformer", "扩散模型", "Muon", "位置编码"].map((q) => createElement("a", { href: hrefFor("/explore", new URLSearchParams({ q })), text: q })),
    ]),
  ]);
  hero.append(createElement("div", { className: "hero-inner" }, [copy, makeHeroVisual(catalog)]));
  const stats = createElement("div", { className: "stat-strip", attrs: { "aria-label": "索引统计" } });
  [[catalog.stats.postCount.toLocaleString("zh-CN"), "篇文章"], [catalog.stats.topicCount, "个主题"], [catalog.stats.seriesCount, "个系列"]].forEach(([value, label]) => stats.append(createElement("span", { className: "stat-item" }, [createElement("strong", { text: value }), ` ${label}`])));
  stats.append(createElement("span", { className: "index-updated", text: `最新文章 ${catalog.stats.latestDate}` }));
  hero.append(stats);
  view.append(hero);

  const shell = createElement("div", { className: "page-shell" });
  const continuation = createElement("div", { id: "continue-reading", className: "continue-reading", attrs: { hidden: true } });
  fillContinueReading(continuation, catalog);
  shell.append(continuation);
  const recent = createElement("section", { className: "home-latest content-section" });
  const articles = createElement("div", { className: "latest-column" }, [createSectionHeading("New in the index", "最近更新", null, { label: "全部文章 ↗", href: "#/explore" })]);
  const list = createElement("div", { className: "post-list" });
  [...catalog.posts].sort((a, b) => dateNumber(b.date) - dateNumber(a.date)).slice(0, 4).forEach((post, index) => {
    const card = makePostCard(post);
    if (index === 0) card.classList.add("is-featured");
    list.append(card);
  });
  articles.append(list);
  recent.append(articles);
  const latestSeries = [...catalog.series].sort((a, b) => dateNumber(b.endDate) - dateNumber(a.endDate))[0];
  if (latestSeries) recent.append(createElement("aside", { className: "latest-series" }, [
    createSectionHeading("An ongoing inquiry", "最近续篇"), makeSeriesCard(latestSeries),
    createElement("p", { className: "margin-note", text: "一个问题，值得用一整个系列慢慢展开。" }),
  ]));
  shell.append(recent);

  const bookshelf = createElement("section", { className: "content-section bookshelf" }, [
    createSectionHeading("Read in sequence", "把思考，读成一本书", "沿着章节顺序，保留推导的来路。", { label: "全部系列 ↗", href: "#/series" }),
  ]);
  const grid = createElement("div", { className: "series-grid" });
  [...catalog.series].sort((a, b) => b.count - a.count).slice(0, 3).forEach((series) => grid.append(makeSeriesCard(series)));
  bookshelf.append(grid);
  if (catalog.series.length) shell.append(bookshelf);
  const topics = createElement("section", { className: "content-section home-topics" }, [
    createSectionHeading("Find your field", "从一个方向出发", null, { label: "探索主题关联 ↗", href: "#/topics" }),
  ]);
  topics.append(makeTopicGroups(catalog));
  shell.append(topics, createElement("section", { className: "library-note" }, [
    createElement("span", { className: "brand-mark", text: "∿", attrs: { "aria-hidden": "true" } }),
    createElement("div", {}, [createElement("h2", { text: "为长久的写作，留一个常新的入口。" }), createElement("p", { text: "人工整理帖容易随时间停更。这个索引持续更新分类与系列，并提供忠于原文的 Markdown，方便非商业学习、保存与提问。阅读与引用始终保留科学空间原文出处。" })]),
    createElement("a", { className: "text-link", href: "#/about", text: "关于索引 ↗" }),
  ]));
  [articles, bookshelf, topics].forEach((section, index) => {
    section.querySelector(".section-heading").dataset.ordinal = String(index + 1).padStart(2, "0");
  });
  view.append(shell, makeFooter());
  return view;
}

function makeTopicGroups(catalog) {
  const groups = createElement("div", { className: "topic-groups" });
  const map = new Map(catalog.topics.map((topic) => [topic.name, topic]));
  catalog.topicGroups.forEach((group, index) => {
    const topics = group.topics.map((name) => map.get(name)).filter((topic) => topic?.count);
    if (!topics.length) return;
    groups.append(createElement("section", { className: "topic-group" }, [
      createElement("span", { className: "group-number", text: String(index + 1).padStart(2, "0") }),
      createElement("h3", { className: "topic-group-title", text: group.name }),
      createElement("div", { className: "topic-grid" }, topics.map(makeTopicCard)),
    ]));
  });
  return groups;
}

function renderTopicIndex(catalog) {
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" }, [
    createElement("section", { className: "page-hero" }, [
      createElement("p", { className: "eyebrow", text: "An atlas of ideas" }),
      createElement("h1", { className: "page-title", text: "知识有方向，也有交汇。" }),
      createElement("p", { className: "page-description", text: `${catalog.stats.topicCount} 个主题，从数学与人工智能，延伸到自然科学与日常思考。` }),
    ]),
    makeTopicGroups(catalog),
  ]);
  const initial = catalog.topics.find((topic) => topic.name === "大模型与Transformer" && topic.count) || catalog.topics.find((topic) => topic.count);
  if (initial) shell.append(makeTopicConnections(catalog, initial.name, { interactive: true }));
  view.append(shell, makeFooter());
  return view;
}

function renderSeriesDetail(catalog, id) {
  const series = catalog.series.find((item) => item.id === id);
  if (!series) return renderNotFound("没有找到这个系列", "它可能还未达到两篇，或系列规则已经更新。", "#/series");
  const posts = postsForSeries(catalog, series);
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero series-detail-hero" }, [
    createElement("div", {}, [
      createElement("a", { className: "back-link", href: "#/series", text: "← 系列书架" }),
      createElement("p", { className: "eyebrow", text: series.topic || "Collected essays" }),
      createElement("h1", { className: "page-title", text: series.name }),
      createElement("p", { className: "page-description", text: `${posts.length} 篇文章${series.startDate && series.endDate ? ` · ${series.startDate.slice(0, 4)} — ${series.endDate.slice(0, 4)}` : ""}。沿章节顺序，读懂一个问题的来龙去脉。` }),
      makeSeriesProgress(series),
    ]),
    makeSeriesCover(series),
  ]));
  const layout = createElement("div", { className: "series-reading-layout" });
  const toc = createElement("details", { className: "chapter-nav", attrs: { open: true } }, [
    createElement("summary", { text: `章节目录 · ${posts.length} 篇` }),
  ]);
  const nav = createElement("nav", { attrs: { "aria-label": "章节目录" } });
  const timeline = createElement("div", { className: "timeline" });
  posts.forEach((post, index) => {
    const chapter = post.seriesIndex !== null ? String(post.seriesIndex).padStart(2, "0") : String(index + 1).padStart(2, "0");
    // Titles stay verbatim in the article; the TOC can omit the repeated series prefix.
    const shortTitle = post.title.startsWith(series.name) ? post.title.slice(series.name.length).replace(/^[（(][^）)]*[）)]\s*[：:、.\s]*/, "").replace(/^[：:、.\s\d]+/, "") || post.title : post.title;
    nav.append(createElement("a", {
      href: seriesChapterHref(series, post), className: isPostRead(post.id) ? "is-read" : "", dataset: { chapterPost: post.id },
      on: { click: (event) => {
        if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        if (event.currentTarget.getAttribute("href") !== location.hash) return;
        event.preventDefault();
        syncSeriesChapter(post.id, { scroll: true, smooth: true });
      } },
    }, [createElement("span", { className: "toc-number", text: chapter }), createElement("span", { text: shortTitle })]));
    const item = createElement("article", {
      className: `timeline-item${isPostRead(post.id) ? " is-read" : ""}`, id: `chapter-${post.id}`, attrs: { tabindex: "-1" }, dataset: { readingPost: post.id },
    });
    item.append(
      createElement("span", { className: "timeline-dot", attrs: { "aria-hidden": "true" } }),
      createElement("div", { className: "timeline-index", text: `${post.seriesIndex !== null ? "CHAPTER" : "ENTRY"} ${chapter}` }),
      createElement("h3", {}, makePostLink(post, item)),
      createElement("div", { className: "post-meta" }, [createElement("span", { text: formatDate(post.date) }), post.level ? createElement("span", { text: LEVEL_LABELS[post.level] ?? post.level }) : null]),
    );
    if (post.sourceSummary) item.append(createElement("p", { className: "post-summary", text: post.sourceSummary }));
    const mirrorActions = makeMirrorActions(post, item);
    if (mirrorActions) item.append(mirrorActions);
    item.append(createElement("div", { className: "timeline-actions" }, [
      makePostProgress(post),
    ]));
    timeline.append(item);
  });
  toc.append(nav);
  layout.append(toc, createElement("section", { className: "chapter-content" }, [createSectionHeading("One chapter at a time", "章节阅读"), timeline]));
  shell.append(layout);
  view.append(shell, makeFooter());
  return view;
}
