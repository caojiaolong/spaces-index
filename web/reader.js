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
  let hideGptSelection = () => {};
  let disposeGpt = () => {};
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
    hideGptSelection();
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
  function containingFormula(node) {
    const formula = (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement)?.closest("mjx-container[data-latex]");
    return formula && articleNode.contains(formula) ? formula : null;
  }
  function formulaTextRange(formula) {
    const math = formula.querySelector("mjx-math");
    if (!math) return null;
    const walker = document.createTreeWalker(math, NodeFilter.SHOW_TEXT, {
      acceptNode: node => node.parentElement.closest("mjx-labels") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
    });
    const texts = [];
    while (walker.nextNode()) if (walker.currentNode.textContent) texts.push(walker.currentNode);
    if (!texts.length) return null;
    const range = document.createRange();
    range.setStart(texts[0], 0);
    range.setEnd(texts.at(-1), texts.at(-1).length);
    return range;
  }
  function textFromRange(range, keepPartial = false) {
    const blocks = new Set("P DIV H1 H2 H3 H4 H5 H6 BLOCKQUOTE PRE UL OL LI TR".split(" "));
    let formulaCount = 0;
    const partialFormulas = [];
    function textOf(node) {
      if (!range.intersectsNode(node)) return "";
      if (node.nodeType === Node.TEXT_NODE) {
        return node.textContent.slice(node === range.startContainer ? range.startOffset : 0,
          node === range.endContainer ? range.endOffset : node.length);
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return "";
      if (node.matches("script, style, template, noscript, [hidden], mjx-assistive-mml, mjx-labels")) return "";
      const style = getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden") return "";
      if (node.matches("mjx-container[data-latex]") && articleNode.contains(node)) {
        formulaCount++;
        const visible = keepPartial && formulaTextRange(node);
        if (visible && (range.compareBoundaryPoints(Range.START_TO_START, visible) > 0 ||
            range.compareBoundaryPoints(Range.END_TO_END, visible) < 0)) {
          const selected = [...node.querySelector("mjx-math").childNodes].map(textOf).join("");
          if (selected) partialFormulas.push({ selected, latex: node.dataset.latex });
          return selected;
        }
        return node.dataset.latex;
      }
      if (node.nodeName === "BR") return "\n";
      const text = [...node.childNodes].map(textOf).join("");
      return blocks.has(node.nodeName) || node.classList?.contains("math-scroll") ? `\n${text}\n` : text;
    }
    const ancestor = containingFormula(range.commonAncestorContainer) || range.commonAncestorContainer;
    return { text: textOf(ancestor).replace(/^\n+|\n+$/g, ""), formulaCount, partialFormulas };
  }
  document.addEventListener("copy", event => {
    if (!current() || !event.clipboardData || articleNode.hidden) return;
    // An input can retain an unrelated document selection while it has focus.
    if (document.activeElement?.matches("input, textarea") || document.activeElement?.isContentEditable) return;
    const selection = getSelection();
    if (!selection) return;
    const focused = document.activeElement?.closest("mjx-container[data-latex]");
    if (selection.isCollapsed && focused && articleNode.contains(focused)) {
      event.clipboardData.setData("text/plain", focused.dataset.latex);
      event.preventDefault(); return;
    }
    if (!selection.rangeCount || selection.isCollapsed) return;
    const range = selection.getRangeAt(0).cloneRange();
    // A selection can start in the title or use the article's parent as its
    // boundary. Check intersection, not whether both endpoints are descendants.
    if (!range.intersectsNode(articleNode)) return;
    const first = containingFormula(range.startContainer);
    const last = containingFormula(range.endContainer);
    // Preserve deliberate character selection within a formula, but copying all
    // its visible text should copy the same source as selecting its container.
    if (first && first === last) {
      // Equation numbers are laid out separately and aren't part of the
      // expression the reader drags across to select the complete formula.
      const visible = formulaTextRange(first);
      if (!visible) return;
      if (range.compareBoundaryPoints(Range.START_TO_START, visible) > 0 ||
          range.compareBoundaryPoints(Range.END_TO_END, visible) < 0) return;
    }
    if (first) range.setStartBefore(first);
    if (last) range.setEndAfter(last);
    const { text, formulaCount } = textFromRange(range);
    if (!formulaCount) return;
    event.clipboardData.setData("text/plain", text);
    event.preventDefault();
  }, { signal });
  function installAi(title, markdown) {
    const markdownUrl = SpacesMirror.publicMarkdownUrl(id, { localPreview: catalog.localPreview });
    const prompt = extra => SpacesMirror.questionPrompt({ id, title, markdownUrl, ...extra });
    const handoff = $("#gpt-handoff"), handoffText = $("#gpt-handoff-text");
    const menu = $("#gpt-options");
    const providerList = $("#ai-provider");
    const menuPanel = $(".reader-gpt-menu"), menuToggle = menu.querySelector("summary");
    let provider = SpacesMirror.preferredAi();
    const providerButtons = [];
    providerList.replaceChildren();
    function providerIcon(item) {
      const icon = document.createElement("span"); icon.className = "ai-provider-icon";
      icon.setAttribute("aria-hidden", "true"); icon.style.setProperty("--ai-icon", `url("${item.icon}")`);
      return icon;
    }
    function labelWithIcon(node, text, item) {
      const label = document.createElement("span"); label.className = "ai-action-label"; label.textContent = text;
      const arrow = document.createElement("span"); arrow.className = "ai-action-arrow";
      arrow.textContent = "↗"; arrow.setAttribute("aria-hidden", "true");
      node.replaceChildren(providerIcon(item), label, arrow);
    }
    for (const item of SpacesMirror.aiProviders) {
      const button = document.createElement("button"); button.type = "button"; button.className = "ai-provider-option";
      button.dataset.aiProvider = item.id; button.setAttribute("role", "radio"); button.setAttribute("aria-label", item.name);
      const name = document.createElement("span"); name.className = "ai-provider-name"; name.textContent = item.label;
      button.append(providerIcon(item), name);
      if (item.id === "chatgpt") {
        const badge = document.createElement("span"); badge.className = "ai-provider-default"; badge.textContent = "默认"; button.append(badge);
      }
      const check = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      check.setAttribute("viewBox", "0 0 16 16"); check.setAttribute("class", "ai-provider-check"); check.setAttribute("aria-hidden", "true");
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", "m3.5 8 3 3 6-6"); check.append(path);
      button.append(check); providerList.append(button); providerButtons.push(button);
      button.addEventListener("click", () => chooseProvider(item.id, true), { signal });
    }
    const popup = document.createElement("div"); popup.className = "selection-actions";
    popup.setAttribute("role", "group"); popup.setAttribute("aria-label", "选段操作"); popup.hidden = true;
    const ask = document.createElement("a"); ask.id = "ask-gpt-selection";
    ask.target = "_blank"; ask.rel = "noopener noreferrer";
    popup.append(ask); document.body.append(popup);
    let timer, snapshot, dragging = false, interacting = false;
    const fullPrompt = prompt({});
    let fullUrl;
    hideGptSelection = () => { clearTimeout(timer); popup.hidden = true; snapshot = null; };
    function refreshProvider() {
      for (const button of providerButtons) {
        const selected = button.dataset.aiProvider === provider.id;
        button.setAttribute("aria-checked", String(selected)); button.tabIndex = selected ? 0 : -1;
      }
      fullUrl = SpacesMirror.aiUrl(fullPrompt, provider.id);
      const full = $("#ask-gpt");
      full.href = fullUrl || provider.url;
      labelWithIcon(full, `全文问 ${provider.label}`, provider);
      full.title = `请 ${provider.name} 获取 Markdown 全文；失败时尝试 GitHub 原始文件，也可复制全文或上传文件`;
      labelWithIcon(ask, `问 ${provider.label}`, provider);
      $("#ai-provider-hint").textContent = provider.query ? `${provider.label} · 链接提问` : `${provider.label} · 复制后粘贴`;
      const heading = $("#gpt-handoff-title"); heading.replaceChildren(providerIcon(provider), document.createTextNode(`向 ${provider.name} 提问`));
      const open = $("#gpt-handoff-open"); open.href = provider.url; labelWithIcon(open, `打开 ${provider.name}`, provider);
      if (menu.open) positionMenu();
    }
    refreshProvider();
    function chooseProvider(identifier, close) {
      provider = SpacesMirror.setPreferredAi(identifier);
      hideGptSelection(); refreshProvider();
      if (close) { menu.open = false; menuToggle.focus({ preventScroll: true }); }
    }
    providerList.addEventListener("keydown", event => {
      const index = providerButtons.indexOf(event.target);
      if (index < 0 || !["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? providerButtons.length - 1 :
        (index + (["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1) + providerButtons.length) % providerButtons.length;
      chooseProvider(providerButtons[next].dataset.aiProvider, false); providerButtons[next].focus({ preventScroll: true });
    }, { signal });
    function positionMenu() {
      if (!menu.open) return;
      const anchor = $(".reader-gpt-actions").getBoundingClientRect();
      const header = document.querySelector("#site-header")?.getBoundingClientRect().bottom || 0;
      const nav = document.querySelector(".main-nav")?.getBoundingClientRect();
      const minTop = Math.max(12, header + 8);
      const maxBottom = nav && nav.top > innerHeight / 2 && nav.bottom >= innerHeight - 30 ? nav.top - 12 : innerHeight - 12;
      if (anchor.bottom < minTop || anchor.top > maxBottom) { menu.open = false; return; }
      menuPanel.style.maxHeight = "";
      const height = menuPanel.getBoundingClientRect().height;
      const below = maxBottom - anchor.bottom - 8, above = anchor.top - minTop - 8;
      const flip = below < height && above > below;
      menuPanel.style.maxHeight = `${Math.max(80, flip ? above : below)}px`;
      menuPanel.style.top = flip ? "auto" : "calc(100% + 8px)";
      menuPanel.style.bottom = flip ? "calc(100% + 8px)" : "auto";
      const width = menuPanel.getBoundingClientRect().width;
      menuPanel.style.left = `${Math.max(12 - anchor.left, Math.min(0, innerWidth - width - 12 - anchor.left))}px`;
    }
    menu.addEventListener("toggle", positionMenu, { signal });
    menu.addEventListener("keydown", event => {
      if (event.key === "Escape" && menu.open) { menu.open = false; menuToggle.focus({ preventScroll: true }); }
    }, { signal });
    addEventListener("scroll", positionMenu, { passive: true, signal });
    addEventListener("resize", positionMenu, { signal });
    function showHandoff(text, note) {
      hideGptSelection(); hideEquationPreview(); menu.open = false;
      handoffText.value = text;
      $("#gpt-handoff-note").textContent = note;
      $("#gpt-handoff-status").textContent = "";
      handoff.showModal(); document.body.classList.add("gpt-handoff-open");
    }
    async function copyHandoff() {
      try {
        await SpacesMirror.copy(handoffText.value);
        if (current() && handoff.open) $("#gpt-handoff-status").textContent = `已复制完整提问内容。打开 ${provider.name} 后粘贴即可。`;
      } catch {
        if (!current() || !handoff.open) return;
        handoffText.focus(); handoffText.select();
        $("#gpt-handoff-status").textContent = "浏览器未允许自动复制。已选中完整内容，请按 Ctrl+C（或 ⌘C），手机上长按复制。";
      }
    }
    function prepareHandoff(text, longNote) {
      showHandoff(text, provider.query ? longNote : `提问已准备好。复制后打开 ${provider.name} 粘贴即可。`);
      if (!provider.query) void copyHandoff();
    }
    $("#ask-gpt").addEventListener("click", event => {
      if (!fullUrl) { event.preventDefault(); prepareHandoff(fullPrompt, `提问链接较长，请复制完整内容后到 ${provider.name} 粘贴。`); }
      menu.open = false;
    }, { signal });
    $("#copy-gpt-full").addEventListener("click", () => {
      showHandoff(prompt({ markdown }), `链接读取失败时，将完整原文粘贴到 ${provider.name}，或下载 Markdown 后上传。这里保留全部署名、来源与许可。`);
      void copyHandoff();
    }, { signal });
    $("#gpt-handoff-copy").addEventListener("click", copyHandoff, { signal });
    $("#gpt-handoff-download").addEventListener("click", () => $("#download-markdown").click(), { signal });
    $("#gpt-handoff-close").addEventListener("click", () => handoff.close(), { signal });
    handoff.addEventListener("close", () => { document.body.classList.remove("gpt-handoff-open"); $("#ask-gpt").focus({ preventScroll: true }); }, { signal });
    function positionPopup() {
      if (!snapshot || popup.hidden) return;
      const margin = 12, header = document.querySelector("#site-header")?.getBoundingClientRect().bottom || 0;
      const nav = document.querySelector(".main-nav")?.getBoundingClientRect();
      const bottom = nav && nav.top > innerHeight / 2 && nav.bottom >= innerHeight - 30 ? nav.top - margin : innerHeight - margin;
      const rects = [...snapshot.range.getClientRects()].filter(r => r.width && r.height && r.bottom > header && r.top < bottom);
      const rect = rects.at(-1);
      if (!rect) { popup.hidden = true; return; }
      const box = popup.getBoundingClientRect();
      let top = rect.bottom + 16;
      if (top + box.height > bottom) top = rect.top - box.height - 16;
      popup.style.left = `${Math.max(margin, Math.min(innerWidth - box.width - margin, rect.left + rect.width / 2 - box.width / 2))}px`;
      popup.style.top = `${Math.max(header + margin, Math.min(bottom - box.height, top))}px`;
    }
    function updateSelection() {
      if (interacting || dragging) return;
      const selection = getSelection();
      if (!current() || articleNode.hidden || handoff.open || imageViewer.open ||
          document.activeElement?.matches("input, textarea") || !selection?.rangeCount || selection.isCollapsed) { hideGptSelection(); return; }
      const range = selection.getRangeAt(0).cloneRange();
      if (!articleNode.contains(range.startContainer) || !articleNode.contains(range.endContainer)) { hideGptSelection(); return; }
      const { text, partialFormulas } = textFromRange(range, true);
      if (!text.trim()) { hideGptSelection(); return; }
      const textPrompt = prompt({ selection: text, partialFormulas });
      snapshot = { range, prompt: textPrompt, url: SpacesMirror.aiUrl(textPrompt, provider.id) };
      ask.href = snapshot.url || provider.url;
      ask.title = partialFormulas.length ? "解释选中字符，并附完整原始公式作为上下文" : "结合 Markdown 全文解释选段";
      hideEquationPreview(); popup.hidden = false; positionPopup();
    }
    function queueSelection() {
      if (interacting) return;
      hideGptSelection();
      timer = setTimeout(updateSelection, 180);
    }
    document.addEventListener("selectionchange", queueSelection, { signal });
    articleNode.addEventListener("pointerdown", () => { dragging = true; hideGptSelection(); }, { signal });
    document.addEventListener("pointerup", event => {
      dragging = false;
      if (!popup.contains(event.target)) interacting = false;
      if (!interacting) queueSelection();
    }, { signal });
    document.addEventListener("pointercancel", () => { dragging = false; interacting = false; queueSelection(); }, { signal });
    ask.addEventListener("pointerdown", event => {
      interacting = true; clearTimeout(timer);
      // Keep mouse selections intact; touch selection menus remain native.
      if (event.pointerType === "mouse") event.preventDefault();
    }, { signal });
    ask.addEventListener("click", event => {
      const saved = snapshot; interacting = false;
      if (!saved) { event.preventDefault(); return; }
      if (!saved.url) { event.preventDefault(); prepareHandoff(saved.prompt, `选段较长，已保留完整内容与原始公式。请复制后打开 ${provider.name} 粘贴，没有截断。`); }
      else hideGptSelection();
    }, { signal });
    document.addEventListener("pointerdown", event => { if (!menu.contains(event.target)) menu.open = false; }, { signal });
    document.addEventListener("keydown", event => { if (event.key === "Escape") { hideGptSelection(); menu.open = false; } }, { signal });
    addEventListener("scroll", positionPopup, { passive: true, signal });
    addEventListener("resize", positionPopup, { signal });
    disposeGpt = () => {
      hideGptSelection(); popup.remove();
      menu.open = false; providerList.replaceChildren();
      if (handoff.open) handoff.close();
      document.body.classList.remove("gpt-handoff-open");
    };
  }
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
      if (!current() || articleNode.hidden || imageViewer.open || $("#gpt-handoff").open) return;
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
    installAi(title, markdown);
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
    disposeGpt();
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
