"use strict";
(function () {
  let mathReady;
  let mathQueue = Promise.resolve();
  function ensureMathJax() {
    if (mathReady) return mathReady;
    const packages = ["base", "action", "ams", "amscd", "bbox", "boldsymbol", "braket", "bussproofs", "cancel", "cases",
      "centernot", "color", "colortbl", "empheq", "enclose", "extpfeil", "gensymb", "html", "mathtools", "mhchem",
      "newcommand", "noerrors", "upgreek", "unicode", "verb", "configmacros", "tagformat", "textcomp", "textmacros"];
    window.MathJax = {
      loader: { paths: { fonts: new URL("./vendor/mathjax/extensions", document.baseURI).href },
        load: [...packages.filter(name => !["base", "ams", "newcommand", "configmacros", "textmacros"].includes(name))
          .map(name => `[tex]/${name}`), "a11y/assistive-mml"] },
      tex: { inlineMath: [["$", "$"], ["\\(", "\\)"]], tags: "ams", processEnvironments: true,
        // Keep the previous extension set, loaded exclusively from local assets.
        // Source \require{AMScd} stays untouched; amscd is already loaded.
        packages,
        macros: { require: ["", 1] }, maxBuffer: 20000 },
      output: { font: "mathjax-tex", fontPath: new URL("./vendor/mathjax/tex-font", document.baseURI).href,
        displayAlign: "center", displayOverflow: "overflow", linebreaks: { inline: false } },
      chtml: { matchFontHeight: false },
      options: { enableMenu: false, enableEnrichment: false, enableSpeech: false, enableBraille: false,
        enableExplorer: false, enableAssistiveMml: true,
        menuOptions: { settings: { enrich: false, speech: false, braille: false, assistiveMml: true } } },
      startup: { typeset: false }
    };

    mathReady = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "./vendor/mathjax/tex-chtml-nofont.js";
      script.onload = () => MathJax.startup.promise.then(resolve, reject);
      script.onerror = () => { mathReady = null; script.remove(); reject(new Error("公式组件加载失败")); };
      document.head.append(script);
    });
    return mathReady;
  }
  function returnRoute(value) {
    return value === "#/" || (typeof value === "string" && /^#\/(?:explore|topics|series)(?:[/?]|$)/.test(value)) ? value : "#/explore";
  }
  function href(id, from) {
    return `#/article/${id}?from=${encodeURIComponent(returnRoute(from))}`;
  }
  function readingPercent({ top, bottom, scroll, viewport, insetTop = 0, insetBottom = 0 }) {
    if (![top, bottom, scroll, viewport, insetTop, insetBottom].every(Number.isFinite) || bottom <= top) return 0;
    const visibleBottom = viewport - insetBottom;
    if (bottom <= visibleBottom + .5 && top < visibleBottom) return 100;
    const start = Math.max(0, top + scroll - insetTop);
    const end = bottom + scroll - visibleBottom;
    // Never round 99.x up to completion before the final line is visible.
    return Math.max(0, Math.min(99, Math.floor((scroll - start) / Math.max(1, end - start) * 100)));
  }
  function mount(view, id, options) {
    const controller = new AbortController();
    const { signal } = controller;
    const catalog = options.catalog;
    const from = returnRoute(options.from);
    const current = () => !signal.aborted && view.isConnected;

  const $ = selector => view.querySelector(selector);
  const status = $("#reader-status");
  const articleNode = $("#article");
  const sourceNode = $("#markdown-source");
  const equationReturn = $("#equation-return");
  const equationPreview = document.createElement("aside");
  equationPreview.id = "equation-preview";
  equationPreview.className = "equation-preview";
  equationPreview.setAttribute("role", "tooltip");
  equationPreview.hidden = true;
  document.body.append(equationPreview);
  let previewOrigin, previewTimer, previewHideTimer;
  function hideEquationPreview() {
    clearTimeout(previewTimer); clearTimeout(previewHideTimer);
    if (previewOrigin?.getAttribute("aria-describedby") === equationPreview.id) previewOrigin.removeAttribute("aria-describedby");
    previewOrigin = null;
    equationPreview.hidden = true;
    equationPreview.replaceChildren();
  }
  function showEquationPreview(link) {
    if (!current() || articleNode.hidden) return;
    const target = document.getElementById(link.dataset.equation);
    if (!target || !articleNode.contains(target)) return;
    const formula = target.closest("mjx-container");
    if (!formula) return;
    hideEquationPreview();
    previewOrigin = link;
    const heading = document.createElement("div"); heading.className = "equation-preview-heading";
    const label = document.createElement("strong"); label.textContent = link.dataset.equationLabel || "公式预览";
    const hint = document.createElement("span"); hint.textContent = "点击编号跳转";
    heading.append(label, hint);
    const content = document.createElement("div"); content.className = "equation-preview-content";
    const clone = formula.cloneNode(true);
    // Reuse the already typeset formula, keeping source labels and the original
    // MathJax document untouched. A preview must never become an anchor target.
    for (const node of [clone, ...clone.querySelectorAll("*")]) {
      for (const name of ["id", "tabindex", "href", "data-equation", "aria-describedby"]) node.removeAttribute(name);
    }
    clone.removeAttribute("aria-label");
    content.append(clone);
    equationPreview.append(heading, content);
    equationPreview.style.width = "";
    equationPreview.hidden = false;
    // Fit ordinary equations without cutting off their right-hand side; long
    // derivations remain scrollable without shrinking the formula's type.
    equationPreview.style.width = `${Math.min(innerWidth - 24, Math.max(320, Math.min(760, clone.scrollWidth + 54)))}px`;
    link.setAttribute("aria-describedby", equationPreview.id);
    const box = link.getBoundingClientRect(), popup = equationPreview.getBoundingClientRect();
    const margin = 12, minTop = (document.querySelector("#site-header")?.getBoundingClientRect().bottom || 0) + margin;
    const below = box.bottom + margin;
    const top = below + popup.height < innerHeight - margin ? below : box.top - popup.height - margin;
    equationPreview.style.left = `${Math.max(margin, Math.min(innerWidth - popup.width - margin, box.left + box.width / 2 - popup.width / 2))}px`;
    equationPreview.style.top = `${Math.max(minTop, Math.min(innerHeight - popup.height - margin, top))}px`;
  }
  function queueEquationPreview(link) {
    clearTimeout(previewTimer); clearTimeout(previewHideTimer);
    previewTimer = setTimeout(() => showEquationPreview(link), 180);
  }
  function leaveEquationPreview() {
    clearTimeout(previewTimer);
    previewHideTimer = setTimeout(hideEquationPreview, 140);
  }
  equationPreview.addEventListener("pointerenter", () => clearTimeout(previewHideTimer), { signal });
  equationPreview.addEventListener("pointerleave", leaveEquationPreview, { signal });
  addEventListener("scroll", hideEquationPreview, { passive: true, signal });
  addEventListener("resize", hideEquationPreview, { signal });
  document.addEventListener("keydown", event => { if (event.key === "Escape") hideEquationPreview(); }, { signal });
  const imageViewer = $("#image-viewer");
  const viewerImage = $("#image-viewer-image");
  const viewerSize = $("#image-viewer-size");
  const viewerStage = $(".image-viewer-stage");
  const viewerZoomIn = $("#image-viewer-zoom-in");
  const viewerZoomOut = $("#image-viewer-zoom-out");
  let imageOrigin;
  let imageOffset = 0;
  let imageScale = 1, fitScale = 1, panX = 0, panY = 0;
  let fittingImage = true;
  let imageDrag;
  const imageReady = () => imageViewer.open && viewerImage.complete && viewerImage.naturalWidth > 0;
  const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));
  function panLimits() {
    return { x: Math.max(0, (viewerImage.naturalWidth * imageScale - viewerStage.clientWidth + 32) / 2),
      y: Math.max(0, (viewerImage.naturalHeight * imageScale - viewerStage.clientHeight + 32) / 2) };
  }
  function paintImage() {
    if (!imageReady()) return;
    const limits = panLimits();
    panX = clamp(panX, limits.x); panY = clamp(panY, limits.y);
    viewerImage.style.transform = `translate(-50%, -50%) translate(${panX}px, ${panY}px) scale(${imageScale})`;
    viewerImage.style.visibility = "visible";
    $("#image-viewer-scale").textContent = `${Math.round(imageScale * 100)}%`;
    viewerZoomOut.disabled = imageScale <= Math.min(.1, fitScale) + .00001;
    viewerZoomIn.disabled = imageScale >= 5 - .00001;
    viewerSize.disabled = false;
    viewerSize.textContent = fittingImage ? "原始尺寸" : "适应窗口";
    viewerSize.setAttribute("aria-pressed", String(!fittingImage));
    viewerStage.classList.toggle("can-pan", limits.x > 0 || limits.y > 0);
  }
  function layoutImage() {
    if (!imageReady()) return;
    fitScale = Math.min(1, Math.max(1, viewerStage.clientWidth - 32) / viewerImage.naturalWidth,
      Math.max(1, viewerStage.clientHeight - 32) / viewerImage.naturalHeight);
    viewerImage.style.width = `${viewerImage.naturalWidth}px`;
    viewerImage.style.height = `${viewerImage.naturalHeight}px`;
    if (fittingImage) { imageScale = fitScale; panX = 0; panY = 0; }
    else imageScale = Math.max(Math.min(.1, fitScale), imageScale);
    paintImage();
  }
  function zoomImage(scale, clientX, clientY) {
    if (!imageReady()) return;
    const next = Math.max(Math.min(.1, fitScale), Math.min(5, scale));
    const box = viewerStage.getBoundingClientRect();
    const x = clientX == null ? 0 : clientX - box.left - box.width / 2;
    const y = clientY == null ? 0 : clientY - box.top - box.height / 2;
    // Keep the image point beneath the pointer stationary while zooming.
    panX = x - (x - panX) * next / imageScale;
    panY = y - (y - panY) * next / imageScale;
    imageScale = next; fittingImage = false;
    paintImage();
  }
  function fitImage() {
    fittingImage = true; layoutImage();
  }
  function endImageDrag() {
    const pointerId = imageDrag?.id;
    imageDrag = null;
    viewerStage.classList.remove("is-dragging");
    if (pointerId != null && viewerStage.hasPointerCapture(pointerId)) viewerStage.releasePointerCapture(pointerId);
  }
  viewerImage.addEventListener("load", layoutImage, { signal });
  viewerImage.addEventListener("error", () => {
    if (imageViewer.open) $("#image-viewer-title").textContent = "图片未能加载";
  }, { signal });
  const imageResize = new ResizeObserver(layoutImage);
  imageResize.observe(viewerStage);
  function openImage(image) {
    imageOrigin = image;
    imageOffset = image.getBoundingClientRect().top;
    fittingImage = true; panX = 0; panY = 0;
    viewerImage.style.visibility = "hidden";
    viewerZoomIn.disabled = viewerZoomOut.disabled = viewerSize.disabled = true;
    $("#image-viewer-scale").textContent = "—";
    $("#image-viewer-title").textContent = "图片预览";
    viewerImage.referrerPolicy = "no-referrer";
    viewerImage.src = image.currentSrc || image.src;
    viewerImage.alt = image.alt;
    $("#image-viewer-original").href = image.dataset.originalSrc || image.src;
    const caption = image.closest(".reader-figure")?.querySelector(".image-caption")?.textContent || image.alt;
    $("#image-viewer-caption").textContent = caption;
    $("#image-viewer-caption").hidden = !caption;
    viewerSize.textContent = "原始尺寸";
    viewerSize.setAttribute("aria-pressed", "false");
    imageViewer.showModal();
    document.body.classList.add("image-viewer-open");
    layoutImage();
  }
  $("#image-viewer-close").addEventListener("click", () => imageViewer.close(), { signal });
  imageViewer.addEventListener("close", () => {
    endImageDrag();
    document.body.classList.remove("image-viewer-open");
    viewerImage.removeAttribute("src");
    if (current() && imageOrigin?.isConnected) {
      imageOrigin.focus({ preventScroll: true });
      scrollTo({ top: scrollY + imageOrigin.getBoundingClientRect().top - imageOffset, behavior: "instant" });
    }
  }, { signal });
  imageViewer.addEventListener("click", event => {
    const box = imageViewer.getBoundingClientRect();
    if (event.target === imageViewer && (event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom)) imageViewer.close();
  }, { signal });
  viewerSize.addEventListener("click", () => {
    if (fittingImage) zoomImage(1);
    else fitImage();
  }, { signal });
  viewerZoomIn.addEventListener("click", () => zoomImage(imageScale * 1.25), { signal });
  viewerZoomOut.addEventListener("click", () => zoomImage(imageScale / 1.25), { signal });
  viewerStage.addEventListener("wheel", event => {
    event.preventDefault();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewerStage.clientHeight : 1);
    zoomImage(imageScale * Math.exp(clamp(-delta * .002, 1)), event.clientX, event.clientY);
  }, { signal, passive: false });
  viewerStage.addEventListener("pointerdown", event => {
    if (!event.isPrimary || event.button !== 0 || !imageReady() || !viewerStage.classList.contains("can-pan")) return;
    event.preventDefault();
    viewerStage.focus({ preventScroll: true });
    imageDrag = { id: event.pointerId, x: event.clientX, y: event.clientY, panX, panY };
    viewerStage.setPointerCapture(event.pointerId);
    viewerStage.classList.add("is-dragging");
  }, { signal });
  viewerStage.addEventListener("pointermove", event => {
    if (!imageDrag || imageDrag.id !== event.pointerId) return;
    panX = imageDrag.panX + event.clientX - imageDrag.x;
    panY = imageDrag.panY + event.clientY - imageDrag.y;
    paintImage();
  }, { signal });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    viewerStage.addEventListener(type, event => {
      if (imageDrag?.id === event.pointerId) endImageDrag();
    }, { signal });
  }
  imageViewer.addEventListener("keydown", event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (["+", "=", "-", "0"].includes(event.key)) {
      event.preventDefault();
      if (event.key === "0") fitImage();
      else zoomImage(imageScale * (event.key === "-" ? .8 : 1.25));
    } else if (event.target === viewerStage && event.key.startsWith("Arrow")) {
      event.preventDefault();
      if (event.key === "ArrowLeft") panX += 40;
      if (event.key === "ArrowRight") panX -= 40;
      if (event.key === "ArrowUp") panY += 40;
      if (event.key === "ArrowDown") panY -= 40;
      paintImage();
    }
  }, { signal });
  articleNode.addEventListener("click", event => {
    if (event.target.tagName !== "IMG" || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); openImage(event.target);
  }, { signal });
  articleNode.addEventListener("keydown", event => {
    if (event.target.tagName !== "IMG" || !["Enter", " "].includes(event.key)) return;
    event.preventDefault(); openImage(event.target);
  }, { signal });
  const equationOrigins = [];
  const mirrorFor = articleId => catalog.posts.find(post => post.id === articleId)?.mirror;
  let readingSize = 18;
  let refreshReadingLayout = () => {};
  let flushReadingProgress = () => {};
  let readingResize;
  let inlineFormulas = [];
  function fitInlineMath() {
    if (!current() || articleNode.hidden) return;
    // Normal inline math keeps MathJax's baseline. Only overlong expressions
    // become scrollable, so clipping never disturbs ordinary subscripts.
    const overflow = inlineFormulas.map(formula => ({ formula,
      wide: formula.querySelector("mjx-math").getBoundingClientRect().width > formula.clientWidth + 1 }));
    for (const { formula, wide } of overflow) formula.classList.toggle("math-inline-overflow", wide);
  }
  const mathResize = new ResizeObserver(fitInlineMath);
  mathResize.observe(articleNode);
  try { readingSize = Math.max(16, Math.min(22, Number(localStorage.getItem("spaces-reader-font")) || 18)); } catch { /* Default size. */ }
  function resizeText(delta) {
    readingSize = Math.max(16, Math.min(22, readingSize + delta));
    articleNode.style.setProperty("--reading-size", `${readingSize}px`);
    try { localStorage.setItem("spaces-reader-font", String(readingSize)); } catch { /* Optional preference. */ }
    requestAnimationFrame(refreshReadingLayout);
  }
  resizeText(0);
  $("#font-smaller").addEventListener("click", () => resizeText(-1));
  $("#font-larger").addEventListener("click", () => resizeText(1));
  const mobileLayout = matchMedia("(max-width: 760px)");
  $("#toc-panel").open = !mobileLayout.matches;
  mobileLayout.addEventListener("change", event => { $("#toc-panel").open = !event.matches; }, { signal });
  function showSource(show) {
    hideEquationPreview();
    articleNode.hidden = show;
    sourceNode.hidden = !show;
    $("#show-reading").setAttribute("aria-pressed", String(!show));
    $("#show-source").setAttribute("aria-pressed", String(show));
    equationReturn.hidden = show || !equationOrigins.length;
    if (show) sourceNode.focus();
    else requestAnimationFrame(refreshReadingLayout);
  }
  let highlightedEquation;
  let highlightTimer;
  const scrollBehavior = () => matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth";
  function highlightEquation(destination) {
    highlightedEquation?.classList.remove("is-equation-target");
    clearTimeout(highlightTimer);
    highlightedEquation = destination;
    destination.classList.add("is-equation-target");
    highlightTimer = setTimeout(() => destination.classList.remove("is-equation-target"), 3000);
  }
  function jumpToEquation(targetId, origin) {
    hideEquationPreview();
    if (!targetId?.startsWith("mjx-eqn:")) return;
    const target = document.getElementById(targetId);
    if (!target || !articleNode.contains(target)) {
      status.textContent = "未找到对应公式，请结合 Markdown 源码或原文阅读。";
      return;
    }
    if (origin) {
      // Keep the reference's viewport offset so returning also survives font
      // changes and late image layout shifts. No extra SPA history entries.
      equationOrigins.push({ link: origin, offset: origin.getBoundingClientRect().top, left: scrollX });
    }
    showSource(false);
    const formula = target.closest("mjx-container");
    const destination = formula?.closest(".math-scroll") || formula || target;
    highlightEquation(destination);
    destination.scrollIntoView({ block: "start", behavior: scrollBehavior() });
    (formula || destination).focus({ preventScroll: true });
    requestAnimationFrame(refreshReadingLayout);
  }
  equationReturn.addEventListener("click", () => {
    const origin = equationOrigins.pop();
    showSource(false);
    if (!origin?.link.isConnected) return;
    const top = scrollY + origin.link.getBoundingClientRect().top - origin.offset;
    origin.link.focus({ preventScroll: true });
    scrollTo({ top, left: origin.left, behavior: scrollBehavior() });
    highlightEquation(origin.link.closest("mjx-container") || origin.link);
    requestAnimationFrame(refreshReadingLayout);
  }, { signal });
  function installMathInteractions() {
    const mathDocument = MathJax.startup.document;
    for (const item of mathDocument.getMathItemsWithin([articleNode])) {
      const formula = item.typesetRoot;
      if (!formula || item.display === null) continue;
      // MathJax retains the unexpanded input and its original delimiters.
      // Never derive copied text from rendered glyphs or rewrite the saved Markdown.
      formula.dataset.latex = item.start.delim + item.math + item.end.delim;
      if (/^\\(?:eqref|ref)\{[^}]+\}$/.test(item.math.trim())) continue;
      formula.tabIndex = 0;
      formula.setAttribute("role", "math");
      formula.setAttribute("aria-label", "公式，按 Ctrl+C（或 ⌘C）复制 LaTeX");
    }
    const labels = Object.values(mathDocument.inputJax[0].parseOptions.tags.allLabels);
    for (const link of articleNode.querySelectorAll("mjx-container a")) {
      const hash = new URL(link.getAttribute("href"), location.href).hash;
      let targetId;
      try { targetId = decodeURIComponent(hash.slice(1)); } catch { continue; }
      if (!targetId.startsWith("mjx-eqn:")) continue;
      const label = labels.find(item => item.id === targetId);
      link.setAttribute("href", href(id, from) + `&equation=${encodeURIComponent(targetId)}`);
      link.setAttribute("aria-label", label ? `跳转到公式 (${label.tag})` : "跳转到对应公式");
      link.dataset.equationLabel = label ? `公式 (${label.tag})` : "公式预览";
      link.removeAttribute("title");
      link.setAttribute("tabindex", "0");
      link.dataset.equation = targetId;
      link.addEventListener("pointerenter", event => { if (event.pointerType === "mouse") queueEquationPreview(link); }, { signal });
      link.addEventListener("pointerleave", leaveEquationPreview, { signal });
      link.addEventListener("focus", () => { if (link.matches(":focus-visible")) queueEquationPreview(link); }, { signal });
      link.addEventListener("blur", hideEquationPreview, { signal });
      link.addEventListener("click", event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        jumpToEquation(targetId, link);
      }, { signal });
    }
  }
  document.addEventListener("copy", event => {
    if (!current() || !event.clipboardData || articleNode.hidden) return;
    const selection = getSelection();
    const focused = document.activeElement?.closest("mjx-container[data-latex]");
    if (selection.isCollapsed && focused && articleNode.contains(focused)) {
      event.clipboardData.setData("text/plain", focused.dataset.latex);
      event.preventDefault(); return;
    }
    if (!selection.rangeCount || selection.isCollapsed) return;
    const range = selection.getRangeAt(0).cloneRange();
    if (!articleNode.contains(range.startContainer) || !articleNode.contains(range.endContainer)) return;
    const containingFormula = node => (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement)?.closest("mjx-container[data-latex]");
    const first = containingFormula(range.startContainer);
    const last = containingFormula(range.endContainer);
    // A drag selection inside one expression is a deliberate selection of its
    // visible characters. Leave native copy intact instead of expanding it.
    if (first && first === last) return;
    if (first) range.setStartBefore(first);
    if (last) range.setEndAfter(last);
    const fragment = range.cloneContents();
    const formulas = fragment.querySelectorAll("mjx-container[data-latex]");
    if (!formulas.length) return;
    for (const formula of formulas) formula.replaceWith(document.createTextNode(formula.dataset.latex));
    const blocks = new Set("P DIV H1 H2 H3 H4 H5 H6 BLOCKQUOTE PRE UL OL LI TR".split(" "));
    function textOf(node) {
      if (node.nodeType === Node.TEXT_NODE) return node.textContent;
      if (node.nodeName === "BR") return "\n";
      const text = [...node.childNodes].map(textOf).join("");
      return blocks.has(node.nodeName) || node.classList?.contains("math-scroll") ? `\n${text}\n` : text;
    }
    event.clipboardData.setData("text/plain", textOf(fragment).replace(/^\n+|\n+$/g, ""));
    event.preventDefault();
  }, { signal });
  function safeUrl(value, mail = false) {
    const url = new URL(value, `https://spaces.ac.cn/archives/${id}`);
    if (!(mail ? ["https:", "http:", "mailto:"] : ["https:", "http:"]).includes(url.protocol) || url.username || url.password) throw new Error("正文链接格式不受支持");
    return url.href;
  }
  const tags = new Set("p div span h1 h2 h3 h4 h5 h6 blockquote strong b em i u del s a img hr br pre code sup sub font center table thead tbody tfoot tr th td caption ul ol li dl dt dd video audio source type".split(" "));
  function render(node, images) {
    if (typeof node === "string") return document.createTextNode(node);
    if (node.tag === "attachment") {
      const details = document.createElement("details"); details.className = "source-attachment";
      const summary = document.createElement("summary"); summary.textContent = `原文嵌入内容（${node.attrs.kind}）`;
      const note = document.createElement("p"); note.textContent = "保留在原文中的位置。此预览不执行脚本或旧插件，可查看原始信息。";
      details.append(summary, note);
      if (node.attrs.url) {
        try {
          const link = document.createElement("a"); link.href = safeUrl(node.attrs.url);
          link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = "查看原始资源"; details.append(link);
        } catch { /* Preserve unusable addresses in the original source below. */ }
      }
      const pre = document.createElement("pre"); const code = document.createElement("code"); code.textContent = node.attrs.source;
      pre.append(code); details.append(pre); return details;
    }
    if (!tags.has(node.tag)) throw new Error("正文包含不受支持的格式");
    const attrs = node.attrs || {};
    if (node.tag === "img") {
      if (!attrs.src) {
        const missing = document.createElement("span"); missing.className = "source-attachment source-missing-image";
        missing.textContent = "［原文图片缺少地址］"; return missing;
      }
      const original = safeUrl(attrs.src);
      const local = images[attrs.src];
      const approved = typeof local === "string" && local.startsWith(`./mirror/${id}/images/`) && /^\.\/mirror\/[1-9]\d*\/images\/[a-f0-9]{64}\.(jpg|png|gif|webp|svg|avif|bmp|ico)$/.test(local);
      const image = document.createElement("img");
      image.src = approved ? local : original;
      image.dataset.originalSrc = original;
      if (approved) image.addEventListener("error", () => { image.src = original; }, { once: true, signal });
      image.alt = attrs.alt || ""; image.loading = "lazy";
      image.referrerPolicy = "no-referrer";
      image.tabIndex = 0;
      image.setAttribute("role", "button");
      image.setAttribute("aria-haspopup", "dialog");
      image.setAttribute("aria-label", image.alt ? `放大图片：${image.alt}` : "放大图片");
      image.title = "点击放大图片";
      return image;
    }
    const element = document.createElement(node.tag);
    if (node.tag === "a" && attrs.href) {
      element.href = safeUrl(attrs.href, true); element.target = "_blank"; element.rel = "noopener noreferrer";
      const original = new URL(element.href);
      const linkedId = original.pathname.match(/^\/archives\/([1-9]\d*)\/?$/)?.[1];
      if (["spaces.ac.cn", "kexue.fm"].includes(original.hostname) && linkedId && mirrorFor(linkedId) && !original.hash) {
        element.href = href(linkedId, from); element.removeAttribute("target");
      }
    }
    for (const key of ["title", "colspan", "rowspan", "start", "value", "color"]) {
      if (attrs[key] !== undefined) element.setAttribute(key, attrs[key]);
    }
    if (attrs.hidden !== undefined) element.hidden = true;
    if (["video", "audio", "source"].includes(node.tag) && attrs.src) element.src = safeUrl(attrs.src);
    if (["video", "audio"].includes(node.tag)) { element.controls = true; element.preload = "none"; }
    for (const child of node.children || []) element.append(render(child, images));
    if (node.tag === "table") {
      const scroll = document.createElement("div"); scroll.className = "table-scroll"; scroll.append(element); return scroll;
    }
    return element;
  }
  function formatImageCaptions(captions) {
    // Match only an adjacent paragraph identified by the verified source HTML.
    // Neither an image's alt text nor the following body paragraph implies a caption.
    const images = [...articleNode.querySelectorAll("img, .source-missing-image")];
    const compact = text => text.replace(/\s+/g, " ").trim();
    for (const { imageIndex, text } of captions) {
      const image = images[imageIndex];
      if (!image || typeof text !== "string") continue;
      let block = image, next;
      while (block !== articleNode) {
        next = block.nextSibling;
        while (next?.nodeType === Node.TEXT_NODE && !next.textContent.trim()) next = next.nextSibling;
        if (next) break;
        block = block.parentElement;
      }
      if (block === articleNode || !next || next.nodeType !== Node.ELEMENT_NODE || next.tagName !== "P" ||
          next.querySelector("img") || compact(next.textContent) !== compact(text)) continue;
      next.classList.add("image-caption");
      next.id = `image-caption-${imageIndex + 1}`;
      image.classList.add("captioned-image");
      image.setAttribute("aria-describedby", next.id);
      // Group the existing nodes without changing their text, links or order.
      const figure = document.createElement("figure"); figure.className = "reader-figure";
      figure.setAttribute("aria-labelledby", next.id);
      block.before(figure);
      figure.append(block, next);
    }
  }
  async function load() {
  try {
    SpacesMirror.path(id);
    const post = catalog.posts.find(item => item.id === id);
    const mirror = post?.mirror;
    if (!mirror) throw new Error("这篇文章的 Markdown 尚未开放或已下架，请从索引访问原文。");
    const response = await fetch(SpacesMirror.path(id, "json"), { cache: "no-store", signal });
    if (!response.ok) throw new Error("Markdown 暂不可用，请阅读原文。");
    const article = await response.json();
    const markdown = await SpacesMirror.verify(article.markdown, mirror.sha256);
    if (!current()) return;
    const title = article.metadata.title;
    $("#reader-title").textContent = title;
    document.title = `${title} · Markdown · 科学空间索引`;
    $("#reader-original").href = `https://spaces.ac.cn/archives/${id}`;
    const topic = post?.topics?.[0];
    $("#reader-date").textContent = article.metadata.date;
    $("#reader-date").dateTime = article.metadata.date;
    $("#reader-number").textContent = `ARTICLE / ${id.padStart(5, "0")}`;
    if (topic) { $("#reader-topic").textContent = topic; $("#reader-topic").href = `#/topics/${encodeURIComponent(topic)}`; }
    if (post?.seriesId) {
      $("#reader-series").hidden = false; $("#reader-series").textContent = `${post.series} ${post.seriesIndex != null ? `· 第 ${post.seriesIndex} 篇` : ""} ↗`;
      $("#reader-series").href = `#/series/${post.seriesId}`;
      const series = catalog.series.find(item => item.id === post.seriesId);
      const position = series?.postIds.indexOf(id) ?? -1;
      for (const [offset, label] of [[-1, "← 上一篇"], [1, "下一篇 →"]]) {
        const nextId = position >= 0 ? series.postIds[position + offset] : null;
        const nextPost = catalog.posts.find(item => item.id === nextId);
        if (!nextPost) continue;
        const link = document.createElement("a"); link.href = mirrorFor(nextId) ? href(nextId, from) : nextPost.url;
        const small = document.createElement("span"); small.textContent = label;
        const title = document.createElement("strong"); title.textContent = nextPost.title;
        link.append(small, title); $("#series-pagination").append(link);
      }
    }
    const fragment = document.createDocumentFragment();
    for (const node of article.tree) fragment.append(render(node, article.images || {}));
    articleNode.replaceChildren(fragment);
    formatImageCaptions(article.captions || []);
    const validationNote = mirror.formulaCount == null ? "正文逐字校验通过；原文公式定界符存在异常。" : `正文与 ${mirror.formulaCount} 段公式通过自动一致性校验。`;
    if (mirror.warnings?.length) {
      const notice = document.createElement("aside"); notice.className = "source-warnings";
      for (const warning of mirror.warnings) { const p = document.createElement("p"); p.textContent = warning; notice.append(p); }
      articleNode.before(notice);
    }
    const attribution = articleNode.querySelector(":scope > blockquote");
    if (attribution) {
      const details = document.createElement("details"); details.className = "reader-attribution";
      const summary = document.createElement("summary"); summary.textContent = "查看完整署名、来源与许可 · CC BY-NC-ND 2.5 CN";
      details.append(summary, attribution); $("#reader-credits").append(details);
    }
    articleNode.querySelector(":scope > h1")?.remove();
    articleNode.querySelector(":scope > hr")?.remove();
    const characters = articleNode.textContent.replace(/\s/g, "").length;
    $("#reader-length").textContent = `约 ${Math.max(1, Math.ceil(characters / 400))} 分钟阅读`;
    const toc = $("#reader-toc"); toc.replaceChildren();
    const headings = [...articleNode.querySelectorAll("h2, h3")];
    headings.forEach((heading, index) => {
      heading.id = `section-${index + 1}`;
      const link = document.createElement("a");
      link.href = href(id, from) + `&section=${heading.id}`;
      link.textContent = heading.textContent;
      link.dataset.target = heading.id;
      link.dataset.depth = heading.tagName.slice(1);
      heading.tabIndex = -1;
      link.addEventListener("click", event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        showSource(false);
        heading.scrollIntoView({ block: "start", behavior: "auto" });
        heading.focus({ preventScroll: true });
        requestAnimationFrame(updateProgress);
      }, { signal });
      toc.append(link);
    });
    if (!headings.length) toc.textContent = "本篇没有章节标题";
    let scrollQueued = false;
    let readingReady = false;
    function updateProgress() {
      scrollQueued = false;
      if (!current() || articleNode.hidden || imageViewer.open) return;
      if (readingReady && document.visibilityState !== "hidden") {
        const box = articleNode.getBoundingClientRect();
        const navigation = document.querySelector(".main-nav")?.getBoundingClientRect();
        const insetBottom = navigation && navigation.top > innerHeight / 2 && navigation.bottom >= innerHeight - 30
          ? innerHeight - navigation.top : 0;
        let progress = readingPercent({ top: box.top, bottom: box.bottom, scroll: scrollY, viewport: innerHeight,
          insetTop: document.querySelector("#site-header")?.getBoundingClientRect().bottom || 0, insetBottom });
        // Lazy images must settle before the end of a shorter provisional layout
        // can count as completion; failed image loads retain their source link.
        if (progress === 100 && [...articleNode.querySelectorAll("img")].some(image => !image.complete)) progress = 99;
        options.onProgress?.(progress);
      }
      $("#reading-percent").textContent = `${options.getProgress?.() || 0}%`;
      const currentHeading = [...headings].reverse().find(heading => heading.getBoundingClientRect().top < 180) || headings[0];
      for (const link of toc.querySelectorAll("a")) {
        const active = link.dataset.target === currentHeading?.id; link.classList.toggle("is-current", active);
        if (active) link.setAttribute("aria-current", "location"); else link.removeAttribute("aria-current");
      }
    }
    refreshReadingLayout = updateProgress;
    flushReadingProgress = updateProgress;
    addEventListener("scroll", () => { if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(updateProgress); } }, { passive: true, signal });
    addEventListener("resize", () => requestAnimationFrame(updateProgress), { signal });
    readingResize = new ResizeObserver(() => requestAnimationFrame(updateProgress));
    readingResize.observe(articleNode);
    articleNode.addEventListener("load", updateProgress, { capture: true, signal });
    articleNode.addEventListener("error", updateProgress, { capture: true, signal });
    addEventListener("pagehide", updateProgress, { signal });
    updateProgress();
    sourceNode.value = markdown;
    $("#reader-actions").hidden = false;
    status.textContent = validationNote + "正在排版公式…";
    $("#show-source").addEventListener("click", () => showSource(true));
    $("#show-reading").addEventListener("click", () => showSource(false));
    $("#copy-markdown").addEventListener("click", async () => {
      try {
        await SpacesMirror.copy(markdown);
        if (current()) $("#reader-copy-status").textContent = "已复制完整 Markdown，包含作者署名、原文链接与许可说明。";
      } catch {
        if (!current()) return;
        showSource(true); sourceNode.select();
        status.textContent = $("#reader-copy-status").textContent = "浏览器未允许自动复制。已选中完整 Markdown，请按 Ctrl+C（或 ⌘C）复制。";
      }
    }, { signal });
    $("#download-markdown").addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
      const link = document.createElement("a"); link.href = url; link.download = `spaces-${id}.md`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    mathQueue = mathQueue.catch(() => {}).then(async () => {
      try {
        await ensureMathJax();
        if (!current()) return;
        MathJax.typesetClear();
        MathJax.texReset();
        await MathJax.typesetPromise([articleNode]);
        await document.fonts.ready;
        if (!current()) return;
        inlineFormulas = [...articleNode.querySelectorAll('mjx-container:not([display="true"])')];
        fitInlineMath();
        for (const formula of articleNode.querySelectorAll('mjx-container[display="true"]')) {
          const scroll = document.createElement("span"); scroll.className = "math-scroll";
          formula.before(scroll); scroll.append(formula);
        }
        installMathInteractions();
        status.textContent = articleNode.querySelector('mjx-merror, [data-mml-node="merror"]')
          ? "部分公式无法排版；Markdown 保留了原始 LaTeX，请结合源码与原文阅读。"
          : validationNote + "公式已排版。";
      } catch {
        if (current()) status.textContent = "公式排版失败，已保留原始 LaTeX，可查看源码或原文。";
      }
      if (!current()) return;
      options.onReady?.(post);
      const section = options.section;
      if (/^section-[1-9]\d*$/.test(section || "")) $("#" + section)?.scrollIntoView({ block: "start" });
      if (options.equation) jumpToEquation(options.equation);
      requestAnimationFrame(() => {
        if (!current()) return;
        readingReady = true;
        updateProgress();
      });
    });
  } catch (error) {
    if (!current()) return;
    $("#reader-title").textContent = "Markdown 暂不可用";
    status.textContent = error.message;
    articleNode.replaceChildren();
    sourceNode.value = "";
    $("#reader-actions").hidden = true;
  }
  }
  const back = $("#reader-back");
  back.href = from;
  back.textContent = from === "#/" ? "返回首页" : from.startsWith("#/series") ? "返回系列目录" : from.startsWith("#/topics") ? "返回主题列表" : "返回文章列表";
  void load();
  return () => {
    flushReadingProgress();
    readingResize?.disconnect();
    hideEquationPreview();
    equationPreview.remove();
    endImageDrag();
    imageResize.disconnect();
    if (imageViewer.open) imageViewer.close();
    document.body.classList.remove("image-viewer-open");
    clearTimeout(highlightTimer);
    mathResize.disconnect();
    controller.abort();
    if (window.MathJax?.typesetClear) MathJax.typesetClear([articleNode]);
  };
  }
  window.SpacesReader = { mount, href, returnRoute, readingPercent };
})();
