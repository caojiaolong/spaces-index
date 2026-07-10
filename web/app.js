const PAGE_SIZE = 24;
const THEME_KEY = "spaces-index-theme";
const READ_POSTS_KEY = "spaces-index-read-posts-v1";
const GITHUB_STARS_CACHE_KEY = "spaces-index-github-stars-v1";
const GITHUB_STARS_CACHE_MS = 30 * 60 * 1000;
const GITHUB_REPOSITORY_API = "https://api.github.com/repos/caojiaolong/spaces-index";
const LEVEL_LABELS = {
  beginner: "入门",
  intermediate: "进阶",
  advanced: "深入",
};

const main = document.querySelector("#main");
const themeButton = document.querySelector("#theme-toggle");
const themeIcon = document.querySelector("#theme-icon");
const themeLabel = document.querySelector("#theme-label");
const header = document.querySelector("#site-header");
const drawerBackdrop = document.querySelector("#drawer-backdrop");
const skipLink = document.querySelector(".skip-link");
const backToTop = document.querySelector("#back-to-top");
const githubLink = document.querySelector("#github-link");
const githubStars = document.querySelector("#github-stars");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const colorScheme = matchMedia("(prefers-color-scheme: dark)");

const ui = {
  catalog: null,
  drawerOpen: false,
  moreFiltersOpen: false,
  focusAfterRender: null,
  reopenSelect: null,
  inputTimer: null,
  lastPath: null,
  scrollAfterRender: false,
  readPostIds: new Set(),
};

function createElement(tag, options = {}, children = []) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined && options.text !== null) {
    element.textContent = String(options.text);
  }
  if (options.id) element.id = options.id;
  if (options.href) element.setAttribute("href", options.href);
  if (options.type) element.setAttribute("type", options.type);
  if (options.value !== undefined) element.value = String(options.value);
  if (options.placeholder) element.setAttribute("placeholder", options.placeholder);
  if (options.attrs) {
    for (const [name, value] of Object.entries(options.attrs)) {
      if (value !== undefined && value !== null && value !== false) {
        element.setAttribute(name, value === true ? "" : String(value));
      }
    }
  }
  if (options.dataset) {
    for (const [name, value] of Object.entries(options.dataset)) {
      element.dataset[name] = String(value);
    }
  }
  if (options.on) {
    for (const [eventName, handler] of Object.entries(options.on)) {
      element.addEventListener(eventName, handler);
    }
  }
  const values = Array.isArray(children) ? children : [children];
  for (const child of values.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function createSvgElement(tag, attrs = {}, children = []) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attrs)) {
    element.setAttribute(name, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child) element.append(child);
  }
  return element;
}

function normalizeText(value) {
  return String(value ?? "").normalize("NFKC").toLocaleLowerCase("zh-CN").trim();
}

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function safeHttpsUrl(value) {
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function dateNumber(value) {
  const parsed = Date.parse(String(value ?? ""));
  return Number.isNaN(parsed) ? 0 : parsed;
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatDate(value) {
  if (!value) return "日期未知";
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})/);
  return match ? `${match[1]} · ${match[2]} · ${match[3]}` : String(value);
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function validStarCount(value) {
  const count = Number(value);
  return Number.isSafeInteger(count) && count >= 0 ? count : null;
}

function showGitHubStarCount(count) {
  if (!githubLink || !githubStars) return;
  const formatted = count.toLocaleString("zh-CN");
  githubStars.textContent = `★ ${formatted}`;
  githubLink.setAttribute("aria-label", `在 GitHub 查看项目，当前 ${formatted} 个 Star`);
  githubLink.title = `GitHub · ${formatted} Stars`;
}

function readCachedGitHubStars() {
  try {
    const cached = JSON.parse(localStorage.getItem(GITHUB_STARS_CACHE_KEY) || "null");
    const count = validStarCount(cached?.count);
    const fetchedAt = Number(cached?.fetchedAt);
    return count === null || !Number.isFinite(fetchedAt) ? null : { count, fetchedAt };
  } catch {
    return null;
  }
}

async function loadGitHubStarCount() {
  const cached = readCachedGitHubStars();
  if (cached) showGitHubStarCount(cached.count);
  if (cached && Date.now() - cached.fetchedAt < GITHUB_STARS_CACHE_MS) return;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(GITHUB_REPOSITORY_API, {
      cache: "no-store",
      headers: { Accept: "application/vnd.github+json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`GitHub API HTTP ${response.status}`);
    const repository = await response.json();
    const count = validStarCount(repository.stargazers_count);
    if (count === null) throw new Error("GitHub API returned an invalid stargazers_count");
    showGitHubStarCount(count);
    try {
      localStorage.setItem(GITHUB_STARS_CACHE_KEY, JSON.stringify({ count, fetchedAt: Date.now() }));
    } catch { /* Storage may be disabled. */ }
  } catch {
    if (!cached && githubStars) githubStars.textContent = "★";
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeCatalog(raw) {
  const rawPosts = Array.isArray(raw.posts) ? raw.posts : [];
  const posts = rawPosts.map((post) => ({
    id: String(post.id ?? ""),
    title: String(post.title ?? "未命名文章"),
    url: String(post.url ?? ""),
    date: String(post.date ?? ""),
    sourceCategory: String(post.sourceCategory ?? post.source_category ?? ""),
    sourceTags: Array.isArray(post.sourceTags ?? post.source_tags)
      ? (post.sourceTags ?? post.source_tags).map(String)
      : [],
    sourceSummary: String(post.sourceSummary ?? post.source_summary ?? ""),
    topics: Array.isArray(post.topics) ? post.topics.map(String) : [],
    series: post.series ? String(post.series) : null,
    seriesId: post.seriesId ?? post.series_id ? String(post.seriesId ?? post.series_id) : null,
    seriesIndex: optionalNumber(post.seriesIndex ?? post.series_index),
    level: post.level ? String(post.level) : null,
    seriesTopic: post.seriesTopic ?? post.series_topic
      ? String(post.seriesTopic ?? post.series_topic)
      : null,
    notes: post.notes ? String(post.notes) : "",
  }));

  const topicCounts = new Map();
  for (const post of posts) {
    for (const topic of post.topics) topicCounts.set(topic, (topicCounts.get(topic) ?? 0) + 1);
  }
  const rawTopics = Array.isArray(raw.topics) ? raw.topics : [];
  const topics = rawTopics.map((topic) => {
    if (typeof topic === "string") return { name: topic, count: topicCounts.get(topic) ?? 0, slug: "" };
    return {
      name: String(topic.name ?? topic.topic ?? ""),
      count: Number(topic.count ?? topic.postCount ?? 0),
      slug: String(topic.slug ?? ""),
    };
  }).filter((topic) => topic.name);
  if (!topics.length) {
    for (const [name, count] of topicCounts) topics.push({ name, count, slug: "" });
  }

  const rawSeries = Array.isArray(raw.series) ? raw.series : [];
  const series = rawSeries.map((item) => ({
    id: String(item.id ?? item.seriesId ?? ""),
    name: String(item.name ?? item.series ?? "未命名系列"),
    topic: String(item.topic ?? item.seriesTopic ?? ""),
    count: Number(item.count ?? item.postCount ?? (Array.isArray(item.postIds) ? item.postIds.length : 0)),
    startDate: String(item.startDate ?? item.dateStart ?? ""),
    endDate: String(item.endDate ?? item.dateEnd ?? ""),
    postIds: Array.isArray(item.postIds ?? item.posts)
      ? (item.postIds ?? item.posts).map((id) => typeof id === "object" ? String(id.id ?? "") : String(id))
      : [],
  })).filter((item) => item.id && item.count >= 2);

  const rawGroups = Array.isArray(raw.topicGroups) ? raw.topicGroups : [];
  const topicGroups = rawGroups.map((group) => ({
    name: String(group.name ?? "主题"),
    topics: Array.isArray(group.topics)
      ? group.topics.map((topic) => typeof topic === "string" ? topic : String(topic.name ?? "")).filter(Boolean)
      : [],
  })).filter((group) => group.topics.length);
  if (!topicGroups.length && topics.length) {
    topicGroups.push({ name: "全部主题", topics: topics.map((topic) => topic.name) });
  }

  const latestPost = [...posts].sort((a, b) => dateNumber(b.date) - dateNumber(a.date))[0];
  const stats = raw.stats ?? {};
  return {
    schemaVersion: Number(raw.schemaVersion ?? 1),
    posts,
    topics,
    topicGroups,
    series,
    stats: {
      postCount: Number(stats.postCount ?? stats.posts ?? posts.length),
      topicCount: Number(stats.topicCount ?? stats.topics ?? topics.filter((topic) => topic.count).length),
      seriesCount: Number(stats.seriesCount ?? stats.series ?? series.length),
      latestDate: String(stats.latestDate ?? stats.latest_date ?? latestPost?.date ?? ""),
    },
  };
}

function parseLocation() {
  const rawHash = location.hash.startsWith("#") ? location.hash.slice(1) : location.hash;
  const [rawPath = "/", query = ""] = rawHash.split("?", 2);
  const path = rawPath.startsWith("/") ? rawPath : `/${rawPath}`;
  return {
    path,
    parts: path.split("/").filter(Boolean).map(safeDecode),
    params: new URLSearchParams(query),
  };
}

function hrefFor(path, params) {
  const query = params?.toString();
  return `#${path}${query ? `?${query}` : ""}`;
}

function setHash(path, params, { replace = false } = {}) {
  const hash = hrefFor(path, params);
  if (replace) {
    history.replaceState(null, "", `${location.pathname}${location.search}${hash}`);
    renderRoute();
  } else if (location.hash === hash) {
    renderRoute();
  } else {
    location.hash = hash;
  }
}

function externalLink(label, url, className = "") {
  const safeUrl = safeHttpsUrl(url);
  if (!safeUrl) return createElement("span", { className, text: label });
  return createElement("a", {
    className,
    text: label,
    href: safeUrl,
    attrs: { target: "_blank", rel: "noopener noreferrer" },
  });
}

function topicHref(name) {
  return `#/topics/${encodeURIComponent(name)}`;
}

function seriesHref(id) {
  return `#/series/${encodeURIComponent(id)}`;
}

function loadReadPostIds() {
  try {
    const stored = JSON.parse(localStorage.getItem(READ_POSTS_KEY) || "[]");
    return new Set(Array.isArray(stored) ? stored.map(String).filter(Boolean) : []);
  } catch {
    return new Set();
  }
}

function persistReadPostIds() {
  try {
    localStorage.setItem(READ_POSTS_KEY, JSON.stringify([...ui.readPostIds].sort()));
  } catch { /* Storage may be disabled. */ }
}

function currentReadPostCount() {
  if (!ui.catalog) return ui.readPostIds.size;
  return ui.catalog.posts.reduce((count, post) => count + Number(isPostRead(post.id)), 0);
}

function syncReadProgressLabels() {
  const count = currentReadPostCount().toLocaleString("zh-CN");
  document.querySelectorAll("[data-read-progress]").forEach((element) => {
    element.textContent = `已读 ${count} / ${ui.catalog?.stats.postCount ?? 0}`;
  });
}

function isPostRead(postId) {
  return ui.readPostIds.has(String(postId));
}

function setPostRead(postId, read) {
  const normalizedId = String(postId);
  if (read) ui.readPostIds.add(normalizedId);
  else ui.readPostIds.delete(normalizedId);
  persistReadPostIds();
  syncReadProgressLabels();
}

function syncReadPresentation(container, post) {
  if (!container) return;
  const read = isPostRead(post.id);
  container.classList.toggle("is-read", read);
  const badge = container.querySelector(".read-status-badge");
  if (badge) badge.hidden = !read;
  const toggle = container.querySelector(".post-read-toggle");
  if (toggle) {
    toggle.textContent = read ? "已读 ✓" : "标为已读";
    toggle.setAttribute("aria-pressed", String(read));
    toggle.setAttribute("aria-label", `${read ? "取消已读" : "标为已读"}：${post.title}`);
  }
}

function updatePostRead(post, read, container, { fromLink = false } = {}) {
  setPostRead(post.id, read);
  const { parts, params } = parseLocation();
  const filteredByReadState = parts[0] === "explore" && ["read", "unread"].includes(params.get("read"));
  if (filteredByReadState) {
    const refresh = () => {
      ui.focusAfterRender = "page-results";
      renderRoute();
    };
    if (fromLink) setTimeout(refresh, 0);
    else refresh();
  } else {
    syncReadPresentation(container, post);
  }
}

function makeReadBadge(post) {
  return createElement("span", {
    className: "read-status-badge",
    text: "已读",
    attrs: { hidden: isPostRead(post.id) ? null : true },
  });
}

function makeReadToggle(post, container) {
  const read = isPostRead(post.id);
  return createElement("button", {
    className: "post-read-toggle",
    type: "button",
    text: read ? "已读 ✓" : "标为已读",
    attrs: {
      "aria-pressed": String(read),
      "aria-label": `${read ? "取消已读" : "标为已读"}：${post.title}`,
    },
    on: { click: () => updatePostRead(post, !isPostRead(post.id), container) },
  });
}

function createSelect({ id, label, value, values, options, multiple = false, onChange }) {
  const selected = new Set(multiple ? values : [value].filter(Boolean));
  const selectedLabels = options.filter((option) => selected.has(option.value)).map((option) => option.label);
  const displayText = selectedLabels.length ? selectedLabels.join("、") : label;
  const root = createElement("div", { className: "select-control", dataset: { selectId: id } });
  const button = createElement("button", {
    className: "select-trigger",
    id: `${id}-trigger`,
    type: "button",
    attrs: {
      role: "combobox",
      "aria-haspopup": "listbox",
      "aria-expanded": "false",
      "aria-controls": `${id}-listbox`,
      "aria-label": selectedLabels.length ? `${label}：${displayText}` : label,
    },
  });
  const buttonText = createElement("span", {
    className: "select-trigger-text",
    text: displayText,
  });
  button.append(buttonText, createElement("span", { className: "select-chevron", text: "▼", attrs: { "aria-hidden": "true" } }));

  const menu = createElement("ul", {
    className: "select-menu",
    id: `${id}-listbox`,
    attrs: {
      role: "listbox",
      "aria-label": label,
      "aria-multiselectable": multiple ? "true" : null,
    },
  });
  const optionNodes = [];
  let activeIndex = 0;

  function close({ focusButton = false } = {}) {
    root.classList.remove("is-open");
    button.setAttribute("aria-expanded", "false");
    button.removeAttribute("aria-activedescendant");
    if (focusButton) button.focus();
  }

  function focusOption(index) {
    if (!optionNodes.length) return;
    activeIndex = (index + optionNodes.length) % optionNodes.length;
    optionNodes.forEach((option, optionIndex) => option.classList.toggle("is-active", optionIndex === activeIndex));
    optionNodes[activeIndex].focus({ preventScroll: true });
    optionNodes[activeIndex].scrollIntoView({ block: "nearest" });
    button.setAttribute("aria-activedescendant", optionNodes[activeIndex].id);
  }

  function open(index) {
    document.querySelectorAll(".select-control.is-open").forEach((openRoot) => {
      if (openRoot !== root && openRoot._closeSelect) openRoot._closeSelect();
    });
    root.classList.add("is-open");
    button.setAttribute("aria-expanded", "true");
    const selectedIndex = optionNodes.findIndex((option) => option.getAttribute("aria-selected") === "true");
    requestAnimationFrame(() => focusOption(index ?? Math.max(0, selectedIndex)));
  }

  function choose(optionValue) {
    if (multiple) {
      const next = new Set(selected);
      if (next.has(optionValue)) next.delete(optionValue);
      else next.add(optionValue);
      ui.reopenSelect = id;
      onChange([...next]);
    } else {
      close();
      ui.focusAfterRender = `${id}-trigger`;
      onChange(optionValue);
    }
  }

  options.forEach((option, index) => {
    const isSelected = selected.has(option.value);
    const item = createElement("li", {
      className: "select-option",
      id: `${id}-option-${index}`,
      attrs: { role: "option", tabindex: "-1", "aria-selected": String(isSelected) },
      on: {
        click: () => choose(option.value),
        keydown: (event) => {
          if (event.key === "ArrowDown") { event.preventDefault(); focusOption(activeIndex + 1); }
          else if (event.key === "ArrowUp") { event.preventDefault(); focusOption(activeIndex - 1); }
          else if (event.key === "Home") { event.preventDefault(); focusOption(0); }
          else if (event.key === "End") { event.preventDefault(); focusOption(optionNodes.length - 1); }
          else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(option.value); }
          else if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close({ focusButton: true });
          }
          else if (event.key === "Tab") close();
        },
      },
    }, [
      createElement("span", { text: option.label }),
      createElement("span", { className: "option-check", text: isSelected ? "✓" : "", attrs: { "aria-hidden": "true" } }),
    ]);
    optionNodes.push(item);
    menu.append(item);
  });

  button.addEventListener("click", () => {
    if (root.classList.contains("is-open")) close();
    else open();
  });
  button.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") { event.preventDefault(); open(0); }
    else if (event.key === "ArrowUp") { event.preventDefault(); open(optionNodes.length - 1); }
    else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!root.classList.contains("is-open")) open();
    } else if (event.key === "Escape" && root.classList.contains("is-open")) {
      event.preventDefault();
      event.stopPropagation();
      close({ focusButton: true });
    }
  });
  root._closeSelect = close;
  root._openSelect = open;
  root.append(button, menu);
  return root;
}

document.addEventListener("pointerdown", (event) => {
  document.querySelectorAll(".select-control.is-open").forEach((root) => {
    if (!root.contains(event.target) && root._closeSelect) root._closeSelect();
  });
});

function makePills(post, limit = 4) {
  const row = createElement("div", { className: "pill-row" });
  for (const topic of post.topics.slice(0, 2)) {
    row.append(createElement("a", { className: "pill", text: topic, href: topicHref(topic) }));
  }
  for (const tag of post.sourceTags.slice(0, Math.max(0, limit - post.topics.slice(0, 2).length))) {
    row.append(createElement("span", { className: "pill", text: `# ${tag}` }));
  }
  return row;
}

function makePostCard(post, { showSummary = true, showActions = false } = {}) {
  const card = createElement("article", {
    className: `post-card${showActions ? " has-actions" : ""}${isPostRead(post.id) ? " is-read" : ""}`,
  });
  const content = createElement("div");
  const safeUrl = safeHttpsUrl(post.url);
  const title = safeUrl
    ? createElement("a", {
        text: post.title,
        href: safeUrl,
        attrs: { target: "_blank", rel: "noopener noreferrer" },
        on: { click: () => updatePostRead(post, true, card, { fromLink: true }) },
      })
    : createElement("span", { text: post.title });
  content.append(createElement("h3", {}, title));

  const meta = createElement("div", { className: "post-meta" }, [
    createElement("span", { text: formatDate(post.date) }),
    post.sourceCategory ? createElement("span", { text: post.sourceCategory }) : null,
    post.seriesId ? createElement("a", {
      text: post.seriesIndex !== null ? `${post.series} · ${post.seriesIndex}` : post.series,
      href: seriesHref(post.seriesId),
    }) : null,
    makeReadBadge(post),
  ]);
  content.append(meta);
  if (showSummary && post.sourceSummary) {
    content.append(createElement("p", { className: "post-summary", text: post.sourceSummary }));
  }
  content.append(createElement("div", { className: "post-card-footer" }, [
    makePills(post),
    makeReadToggle(post, card),
  ]));
  if (showActions) {
    const actions = createElement("div", { className: "post-card-actions" });
    if (safeUrl) {
      actions.append(createElement("a", {
        className: "post-card-action",
        text: "阅读原文 ↗",
        href: safeUrl,
        attrs: { target: "_blank", rel: "noopener noreferrer" },
        on: { click: () => updatePostRead(post, true, card, { fromLink: true }) },
      }));
    }
    if (post.seriesId) {
      actions.append(createElement("a", {
        className: "post-card-action series-action",
        text: "查看系列 →",
        href: seriesHref(post.seriesId),
      }));
    }
    if (actions.childElementCount) content.append(actions);
  }
  card.append(content, createElement("span", { className: "post-arrow", text: "↗", attrs: { "aria-hidden": "true" } }));
  return card;
}

function makeSeriesCard(series) {
  return createElement("a", { className: "series-card", href: seriesHref(series.id) }, [
    createElement("span", { className: "card-kicker", text: series.topic || "专题系列" }),
    createElement("h3", { text: series.name }),
    createElement("div", { className: "series-meta" }, [
      createElement("span", { text: `${series.count} 篇` }),
      series.startDate && series.endDate
        ? createElement("span", { text: `${series.startDate.slice(0, 4)} — ${series.endDate.slice(0, 4)}` })
        : null,
    ]),
  ]);
}

function makeTopicCard(topic) {
  return createElement("a", { className: "topic-card", href: topicHref(topic.name) }, [
    createElement("h3", { text: topic.name }),
    createElement("span", { text: `${topic.count} ↗` }),
  ]);
}

function makeFooter() {
  const shell = createElement("div", { className: "page-shell" });
  const footer = createElement("footer", { className: "site-footer" }, [
    createElement("div", {}, [
      createElement("strong", { text: "科学空间索引" }),
      createElement("div", { text: "非官方元数据导航 · 不保存文章正文" }),
    ]),
    createElement("div", { className: "site-footer-links" }, [
      externalLink("原站", "https://spaces.ac.cn/"),
      externalLink("GitHub", "https://github.com/caojiaolong/spaces-index"),
      createElement("a", { text: "关于", href: "#/about" }),
    ]),
  ]);
  shell.append(footer);
  return shell;
}

function makeHeroVisual() {
  const gradient = createSvgElement("linearGradient", { id: "curve-gradient", x1: "0", y1: "0", x2: "1", y2: "1" }, [
    createSvgElement("stop", { offset: "0", "stop-color": "#49a8ff" }),
    createSvgElement("stop", { offset: ".5", "stop-color": "#8d7dff" }),
    createSvgElement("stop", { offset: "1", "stop-color": "#c27be8" }),
  ]);
  const svg = createSvgElement("svg", {
    class: "hero-svg",
    viewBox: "0 0 520 520",
    role: "img",
    "aria-label": "由概率轨道、矩阵节点和损失曲线构成的数学图景",
  }, [
    createSvgElement("defs", {}, [gradient]),
    createSvgElement("circle", { class: "orbit-line", cx: "260", cy: "260", r: "190" }),
    createSvgElement("ellipse", { class: "orbit-line", cx: "260", cy: "260", rx: "208", ry: "86", transform: "rotate(-28 260 260)" }),
    createSvgElement("ellipse", { class: "orbit-line", cx: "260", cy: "260", rx: "208", ry: "86", transform: "rotate(34 260 260)" }),
    createSvgElement("path", { class: "curve-glow", d: "M37 335 C108 326 114 137 195 157 S273 376 345 310 S390 114 489 147" }),
    createSvgElement("path", { class: "curve-line", d: "M37 335 C108 326 114 137 195 157 S273 376 345 310 S390 114 489 147" }),
    createSvgElement("circle", { class: "node", cx: "108", cy: "257", r: "5" }),
    createSvgElement("circle", { class: "node", cx: "195", cy: "157", r: "6" }),
    createSvgElement("circle", { class: "node", cx: "260", cy: "259", r: "6" }),
    createSvgElement("circle", { class: "node", cx: "345", cy: "310", r: "5" }),
    createSvgElement("circle", { class: "node", cx: "430", cy: "157", r: "6" }),
    createSvgElement("text", { class: "visual-label", x: "42", y: "374" }, []),
    createSvgElement("text", { class: "visual-label", x: "365", y: "445" }, []),
  ]);
  svg.querySelectorAll("text")[0].textContent = "∂L / ∂θ → 0";
  svg.querySelectorAll("text")[1].textContent = "p(z|x) ∝ exp(−E)";
  return createElement("div", { className: "math-visual", attrs: { "aria-hidden": "false" } }, svg);
}

function createSectionHeading(eyebrow, title, description, link) {
  const left = createElement("div", {}, [
    createElement("p", { className: "eyebrow", text: eyebrow }),
    createElement("h2", { text: title }),
    description ? createElement("p", { text: description }) : null,
  ]);
  return createElement("div", { className: "section-heading" }, [
    left,
    link ? createElement("a", { className: "text-link", text: link.label, href: link.href }) : null,
  ]);
}

function renderHome(catalog) {
  const view = createElement("div", { className: "view" });
  const hero = createElement("section", { className: "hero" });
  const heroCopy = createElement("div", {}, [
    createElement("p", { className: "eyebrow", text: "Unofficial metadata atlas" }),
    createElement("h1", {}, [
      "让散落的思考，\n",
      createElement("span", { className: "gradient-text", text: "形成可探索的坐标。" }),
    ]),
    createElement("p", {
      className: "hero-copy",
      text: "为科学空间文章建立一张持续生长的知识地图。沿着主题、系列、标签与时间，重新发现数学和人工智能中的长线思考。",
    }),
    createElement("div", { className: "hero-actions" }, [
      createElement("a", { className: "button button-primary", text: "开始探索  →", href: "#/explore" }),
      externalLink("访问科学空间  ↗", "https://spaces.ac.cn/", "button button-ghost"),
    ]),
    createElement("p", { className: "hero-note", text: "仅索引公开元数据与短小结，不镜像、不保存文章正文。" }),
  ]);
  hero.append(createElement("div", { className: "hero-inner" }, [heroCopy, makeHeroVisual()]));
  view.append(hero);

  const stats = [
    [catalog.stats.postCount, "收录文章"],
    [catalog.stats.topicCount, "知识主题"],
    [catalog.stats.seriesCount, "连续系列"],
    [catalog.stats.latestDate, "最近更新"],
  ];
  const statStrip = createElement("section", { className: "stat-strip", attrs: { "aria-label": "索引统计" } });
  stats.forEach(([value, label], index) => {
    statStrip.append(createElement("div", { className: "stat-item" }, [
      createElement("span", {
        className: "stat-value",
        text: index < 3 && !reduceMotion.matches ? "0" : value,
        dataset: index < 3 ? { countTarget: value } : {},
      }),
      createElement("span", { className: "stat-label", text: label }),
    ]));
  });
  view.append(statStrip);

  const story = createElement("section", { className: "content-section" });
  const storyShell = createElement("div", { className: "page-shell" });
  storyShell.append(createSectionHeading(
    "Why this index",
    "把漫长写作，读成一条思想的轨迹",
    "索引不替代原文。它只是给持续十余年的知识积累，补上一层更容易进入的导航。",
  ));
  const storyGrid = createElement("div", { className: "story-grid" });
  [
    ["01", "从一个问题出发", "用标题、主题与短小结快速判断一篇文章是否值得深入。", "q → ?"],
    ["02", "沿系列连续阅读", "自动识别文章系列和章节顺序，让上下文不再散落。", "xₜ → xₜ₊₁"],
    ["03", "在主题之间漫游", "从 Transformer 到几何方程，看见概念之间意外的连接。", "AᵀA ≽ 0"],
  ].forEach(([index, title, copy, formula]) => {
    storyGrid.append(createElement("article", { className: "story-card" }, [
      createElement("span", { className: "story-index", text: index }),
      createElement("span", { className: "story-formula", text: formula, attrs: { "aria-hidden": "true" } }),
      createElement("h3", { text: title }),
      createElement("p", { text: copy }),
    ]));
  });
  storyShell.append(storyGrid);
  story.append(storyShell);
  view.append(story);

  const recent = createElement("section", { className: "content-section compact" });
  const recentShell = createElement("div", { className: "page-shell" });
  recentShell.append(createSectionHeading(
    "Latest signals",
    "最近更新",
    "新的问题、新的推导，以及仍在继续的系列。",
    { label: "查看全部文章 →", href: "#/explore" },
  ));
  const recentList = createElement("div", { className: "post-list" });
  [...catalog.posts].sort((a, b) => dateNumber(b.date) - dateNumber(a.date)).slice(0, 6)
    .forEach((post) => recentList.append(makePostCard(post, { showSummary: true, showActions: true })));
  recentShell.append(recentList);
  recent.append(recentShell);
  view.append(recent);

  const topicSection = createElement("section", { className: "content-section compact" });
  const topicShell = createElement("div", { className: "page-shell" });
  topicShell.append(createSectionHeading(
    "Knowledge fields",
    "从主题进入",
    "这些不是封闭的抽屉，而是互相交叠的知识坐标。",
    { label: "浏览主题地图 →", href: "#/topics" },
  ));
  const topicGrid = createElement("div", { className: "topic-grid" });
  [...catalog.topics].sort((a, b) => b.count - a.count).slice(0, 12).forEach((topic) => topicGrid.append(makeTopicCard(topic)));
  topicShell.append(topicGrid);
  topicSection.append(topicShell);
  view.append(topicSection);

  const longSeries = [...catalog.series].sort((a, b) => b.count - a.count).slice(0, 6);
  if (longSeries.length) {
    const seriesSection = createElement("section", { className: "content-section compact" });
    const seriesShell = createElement("div", { className: "page-shell" });
    seriesShell.append(createSectionHeading(
      "Long-form thinking",
      "值得连续阅读的系列",
      "有些问题需要多篇文章，才能慢慢展开它真正的形状。",
      { label: "查看全部系列 →", href: "#/series" },
    ));
    const grid = createElement("div", { className: "series-grid" });
    longSeries.forEach((series) => grid.append(makeSeriesCard(series)));
    seriesShell.append(grid);
    seriesSection.append(seriesShell);
    view.append(seriesSection);
  }
  view.append(makeFooter());
  return view;
}

function animateHomeStats() {
  if (reduceMotion.matches) return;
  document.querySelectorAll("[data-count-target]").forEach((element) => {
    const target = Number(element.dataset.countTarget);
    if (!Number.isFinite(target)) return;
    const start = performance.now();
    const duration = 700;
    function frame(now) {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = Math.round(target * eased).toLocaleString("zh-CN");
      if (progress < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
}

function readExploreState(params) {
  let topics = params.getAll("topic").map(safeDecode).filter(Boolean);
  if (!topics.length && params.get("topics")) {
    topics = params.get("topics").split(",").map(safeDecode).filter(Boolean);
  }
  const series = params.get("series");
  const read = params.get("read");
  return {
    q: params.get("q") ?? "",
    topics: unique(topics),
    tag: params.get("tag") ?? "",
    category: params.get("category") ?? "",
    year: params.get("year") ?? "",
    series: ["1", "series"].includes(series) ? "series" : (["0", "standalone"].includes(series) ? "standalone" : ""),
    read: ["read", "unread"].includes(read) ? read : "",
    level: params.get("level") ?? "",
    sort: params.get("sort") ?? (params.get("q") ? "relevance" : "latest"),
    page: Math.max(1, Number.parseInt(params.get("page") ?? "1", 10) || 1),
  };
}

function exploreParams(state) {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  state.topics.forEach((topic) => params.append("topic", topic));
  if (state.tag) params.set("tag", state.tag);
  if (state.category) params.set("category", state.category);
  if (state.year) params.set("year", state.year);
  if (state.series === "series") params.set("series", "1");
  else if (state.series === "standalone") params.set("series", "0");
  if (state.read) params.set("read", state.read);
  if (state.level) params.set("level", state.level);
  const defaultSort = state.q ? "relevance" : "latest";
  if (state.sort && state.sort !== defaultSort) params.set("sort", state.sort);
  if (state.page > 1) params.set("page", String(state.page));
  return params;
}

function updateExplore(state, patch, { replace = false, focus } = {}) {
  clearTimeout(ui.inputTimer);
  ui.inputTimer = null;
  const next = { ...state, ...patch };
  if (!("page" in patch)) next.page = 1;
  if (focus !== undefined) ui.focusAfterRender = focus;
  setHash("/explore", exploreParams(next), { replace });
}

function searchAndFilter(catalog, state) {
  const tokens = normalizeText(state.q).split(/\s+/).filter(Boolean);
  const normalizedTag = normalizeText(state.tag);
  const scored = [];

  for (const post of catalog.posts) {
    if (state.topics.length && !post.topics.some((topic) => state.topics.includes(topic))) continue;
    if (normalizedTag && !post.sourceTags.some((tag) => normalizeText(tag).includes(normalizedTag))) continue;
    if (state.category && post.sourceCategory !== state.category) continue;
    if (state.year && !post.date.startsWith(`${state.year}-`)) continue;
    if (state.series === "series" && !post.seriesId) continue;
    if (state.series === "standalone" && post.seriesId) continue;
    if (state.read === "read" && !isPostRead(post.id)) continue;
    if (state.read === "unread" && isPostRead(post.id)) continue;
    if (state.level && post.level !== state.level) continue;

    const fields = {
      title: normalizeText(post.title),
      series: normalizeText(post.series),
      tags: normalizeText(post.sourceTags.join(" ")),
      summary: normalizeText(post.sourceSummary),
    };
    let score = 0;
    let matchesEveryToken = true;
    for (const token of tokens) {
      let tokenScore = 0;
      if (fields.title.includes(token)) tokenScore += 5;
      if (fields.series.includes(token)) tokenScore += 3;
      if (fields.tags.includes(token)) tokenScore += 2;
      if (fields.summary.includes(token)) tokenScore += 1;
      if (!tokenScore) {
        matchesEveryToken = false;
        break;
      }
      score += tokenScore;
    }
    if (matchesEveryToken) scored.push({ post, score });
  }

  const sort = state.sort;
  scored.sort((left, right) => {
    if (sort === "relevance" && right.score !== left.score) return right.score - left.score;
    if (sort === "oldest") return dateNumber(left.post.date) - dateNumber(right.post.date);
    if (sort === "title") return left.post.title.localeCompare(right.post.title, "zh-CN");
    return dateNumber(right.post.date) - dateNumber(left.post.date);
  });
  return scored.map((entry) => entry.post);
}

function openFilterDrawer() {
  ui.drawerOpen = true;
  document.body.classList.add("drawer-open");
  drawerBackdrop.hidden = false;
  const panel = document.querySelector("#filters-panel");
  panel?.classList.add("is-open");
  panel?.setAttribute("aria-hidden", "false");
  panel?.removeAttribute("inert");
  if (panel) panel.scrollTop = 0;
  document.querySelector("#mobile-filter-button")?.setAttribute("aria-expanded", "true");
  requestAnimationFrame(() => panel?.querySelector("button")?.focus());
}

function closeFilterDrawer({ restoreFocus = true } = {}) {
  ui.drawerOpen = false;
  document.body.classList.remove("drawer-open");
  drawerBackdrop.hidden = true;
  const panel = document.querySelector("#filters-panel");
  panel?.classList.remove("is-open");
  panel?.setAttribute("aria-hidden", "true");
  panel?.setAttribute("inert", "");
  document.querySelector("#mobile-filter-button")?.setAttribute("aria-expanded", "false");
  if (restoreFocus) document.querySelector("#mobile-filter-button")?.focus();
}

function makeFilterChip(label, onRemove) {
  return createElement("button", {
    className: "filter-chip",
    type: "button",
    attrs: { "aria-label": `移除筛选：${label}` },
    on: { click: onRemove },
  }, [createElement("span", { text: label }), createElement("span", { text: "×", attrs: { "aria-hidden": "true" } })]);
}

function makePagination({ current, total, href }) {
  if (total <= 1) return null;
  const prepareNavigation = (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    ui.scrollAfterRender = true;
  };
  const nav = createElement("nav", { className: "pagination", attrs: { "aria-label": "分页" } });
  const previous = createElement("a", {
    className: "page-button",
    text: "←",
    href: current > 1 ? href(current - 1) : href(1),
    attrs: { "aria-label": "上一页", "aria-disabled": current <= 1 ? "true" : null },
  });
  if (current <= 1) previous.addEventListener("click", (event) => event.preventDefault());
  else previous.addEventListener("click", prepareNavigation);
  nav.append(previous);

  const pages = unique([1, current - 1, current, current + 1, total]).filter((page) => page >= 1 && page <= total).sort((a, b) => a - b);
  let previousPage = 0;
  for (const page of pages) {
    if (previousPage && page - previousPage > 1) {
      nav.append(createElement("span", { className: "page-button", text: "…", attrs: { "aria-hidden": "true" } }));
    }
    const pageLink = createElement("a", {
      className: "page-button",
      text: page,
      href: href(page),
      attrs: { "aria-label": `第 ${page} 页`, "aria-current": page === current ? "page" : null },
    });
    if (page !== current) pageLink.addEventListener("click", prepareNavigation);
    nav.append(pageLink);
    previousPage = page;
  }

  const next = createElement("a", {
    className: "page-button",
    text: "→",
    href: current < total ? href(current + 1) : href(total),
    attrs: { "aria-label": "下一页", "aria-disabled": current >= total ? "true" : null },
  });
  if (current >= total) next.addEventListener("click", (event) => event.preventDefault());
  else next.addEventListener("click", prepareNavigation);
  nav.append(next);
  return nav;
}

function renderExplore(catalog, params) {
  const state = readExploreState(params);
  const results = searchAndFilter(catalog, state);
  const totalPages = Math.max(1, Math.ceil(results.length / PAGE_SIZE));
  const currentPage = Math.min(state.page, totalPages);
  const pageResults = results.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const view = createElement("div", { className: "view" });

  const heroShell = createElement("div", { className: "page-shell" });
  heroShell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: "Explore the archive" }),
    createElement("h1", { className: "page-title", text: "在知识坐标里，寻找下一篇文章" }),
    createElement("p", { className: "page-description", text: "输入问题或概念，也可以按主题、系列、标签和时间逐层收窄。" }),
  ]));
  view.append(heroShell);

  const layout = createElement("div", { className: "page-shell explore-layout" });
  const panel = createElement("aside", {
    className: `filters-panel${innerWidth <= 760 && ui.drawerOpen ? " is-open" : ""}`,
    id: "filters-panel",
    attrs: {
      "aria-label": "文章筛选",
      "aria-hidden": innerWidth <= 760 && !ui.drawerOpen ? "true" : "false",
      role: innerWidth <= 760 ? "dialog" : "region",
      "aria-modal": innerWidth <= 760 ? "true" : null,
      inert: innerWidth <= 760 && !ui.drawerOpen ? true : null,
    },
  });
  const heading = createElement("div", { className: "filter-heading" }, [
    createElement("h2", { text: "筛选条件" }),
    createElement("button", {
      className: "filter-close",
      type: "button",
      attrs: { "aria-label": "关闭筛选" },
      text: "×",
      on: { click: () => closeFilterDrawer() },
    }),
  ]);
  panel.append(heading);

  const topicOptions = [...catalog.topics]
    .filter((topic) => topic.count)
    .map((topic) => ({ value: topic.name, label: `${topic.name} · ${topic.count}` }));
  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("span", { className: "filter-label", text: "主题（可多选）", id: "topic-filter-label" }),
    createSelect({
      id: "topic-filter",
      label: "全部主题",
      values: state.topics,
      options: topicOptions,
      multiple: true,
      onChange: (topics) => updateExplore(state, { topics }),
    }),
  ]));

  const tags = unique(catalog.posts.flatMap((post) => post.sourceTags)).sort((a, b) => a.localeCompare(b, "zh-CN"));
  const tagInput = createElement("input", {
    className: "text-input",
    id: "tag-filter",
    type: "search",
    value: state.tag,
    placeholder: "输入或选择标签",
    attrs: { list: "tag-options", autocomplete: "off" },
    on: {
      input: (event) => {
        if (event.isComposing) return;
        clearTimeout(ui.inputTimer);
        const tag = event.currentTarget.value;
        ui.inputTimer = setTimeout(() => updateExplore(state, { tag }, { replace: true, focus: "tag-filter" }), 300);
      },
      compositionstart: () => clearTimeout(ui.inputTimer),
      compositionend: (event) => {
        clearTimeout(ui.inputTimer);
        const tag = event.currentTarget.value;
        ui.inputTimer = setTimeout(() => updateExplore(state, { tag }, { replace: true, focus: "tag-filter" }), 300);
      },
      keydown: (event) => {
        if (event.key === "Enter" && !event.isComposing) {
          event.preventDefault();
          clearTimeout(ui.inputTimer);
          updateExplore(state, { tag: event.currentTarget.value }, { focus: "tag-filter" });
        }
      },
    },
  });
  const datalist = createElement("datalist", { id: "tag-options" });
  tags.forEach((tag) => datalist.append(createElement("option", { value: tag })));
  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("label", { className: "filter-label", text: "标签", attrs: { for: "tag-filter" } }),
    tagInput,
    datalist,
  ]));

  const categories = unique(catalog.posts.map((post) => post.sourceCategory)).sort((a, b) => a.localeCompare(b, "zh-CN"));
  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("span", { className: "filter-label", text: "原站分类" }),
    createSelect({
      id: "category-filter",
      label: "全部分类",
      value: state.category,
      options: [{ value: "", label: "全部分类" }, ...categories.map((category) => ({ value: category, label: category }))],
      onChange: (category) => updateExplore(state, { category }),
    }),
  ]));

  const years = unique(catalog.posts.map((post) => post.date.slice(0, 4)).filter((year) => /^\d{4}$/.test(year))).sort((a, b) => b.localeCompare(a));
  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("span", { className: "filter-label", text: "年份" }),
    createSelect({
      id: "year-filter",
      label: "全部年份",
      value: state.year,
      options: [{ value: "", label: "全部年份" }, ...years.map((year) => ({ value: year, label: year }))],
      onChange: (year) => updateExplore(state, { year }),
    }),
  ]));

  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("span", { className: "filter-label", text: "系列" }),
    createSelect({
      id: "series-filter",
      label: "全部文章",
      value: state.series,
      options: [
        { value: "", label: "全部文章" },
        { value: "series", label: "仅看系列文章" },
        { value: "standalone", label: "仅看非系列文章" },
      ],
      onChange: (series) => updateExplore(state, { series }),
    }),
  ]));

  panel.append(createElement("div", { className: "filter-group" }, [
    createElement("span", { className: "filter-label", text: "阅读状态" }),
    createSelect({
      id: "read-filter",
      label: "全部阅读状态",
      value: state.read,
      options: [
        { value: "", label: "全部阅读状态" },
        { value: "read", label: "只看已读" },
        { value: "unread", label: "只看未读" },
      ],
      onChange: (read) => updateExplore(state, { read }),
    }),
  ]));

  const more = createElement("div", { className: `more-filters${ui.moreFiltersOpen || state.level ? " is-open" : ""}` });
  const moreButton = createElement("button", {
    className: "more-toggle",
    type: "button",
    attrs: { "aria-expanded": String(ui.moreFiltersOpen || Boolean(state.level)), "aria-controls": "more-filter-content" },
    on: {
      click: (event) => {
        ui.moreFiltersOpen = !event.currentTarget.closest(".more-filters").classList.contains("is-open");
        event.currentTarget.closest(".more-filters").classList.toggle("is-open", ui.moreFiltersOpen);
        event.currentTarget.setAttribute("aria-expanded", String(ui.moreFiltersOpen));
      },
    },
  }, [createElement("span", { text: "更多筛选" }), createElement("span", { text: "+", attrs: { "aria-hidden": "true" } })]);
  const moreContent = createElement("div", { className: "more-filter-content", id: "more-filter-content" }, [
    createElement("div", { className: "filter-group" }, [
      createElement("span", { className: "filter-label", text: "阅读深度" }),
      createSelect({
        id: "level-filter",
        label: "全部难度",
        value: state.level,
        options: [
          { value: "", label: "全部难度" },
          { value: "beginner", label: "入门" },
          { value: "intermediate", label: "进阶" },
          { value: "advanced", label: "深入" },
        ],
        onChange: (level) => updateExplore(state, { level }),
      }),
    ]),
  ]);
  more.append(moreButton, moreContent);
  panel.append(more);

  panel.append(createElement("button", {
    className: "filter-reset",
    type: "button",
    text: "清除全部条件",
    on: { click: () => setHash("/explore", new URLSearchParams()) },
  }));
  panel.append(createElement("button", {
    className: "button button-primary filter-apply",
    type: "button",
    text: `查看 ${results.length} 篇文章`,
    on: { click: () => closeFilterDrawer() },
  }));

  const resultsColumn = createElement("section", {
    id: "page-results",
    attrs: { "aria-label": "搜索结果", tabindex: "-1" },
  });
  const searchInput = createElement("input", {
    className: "search-input",
    id: "explore-search",
    type: "search",
    value: state.q,
    placeholder: "搜索标题、系列、标签或小结…",
    attrs: { autocomplete: "off", "aria-label": "搜索文章" },
    on: {
      input: (event) => {
        if (event.isComposing) return;
        clearTimeout(ui.inputTimer);
        const q = event.currentTarget.value;
        ui.inputTimer = setTimeout(() => updateExplore(state, { q, sort: q ? "relevance" : "latest" }, { replace: true, focus: "explore-search" }), 220);
      },
      compositionstart: () => clearTimeout(ui.inputTimer),
      compositionend: (event) => {
        clearTimeout(ui.inputTimer);
        const q = event.currentTarget.value;
        ui.inputTimer = setTimeout(() => updateExplore(state, { q, sort: q ? "relevance" : "latest" }, { replace: true, focus: "explore-search" }), 220);
      },
      keydown: (event) => {
        if (event.key === "Enter" && !event.isComposing) {
          event.preventDefault();
          clearTimeout(ui.inputTimer);
          const q = event.currentTarget.value;
          updateExplore(state, { q, sort: q ? "relevance" : "latest" }, { focus: "explore-search" });
        }
      },
    },
  });
  resultsColumn.append(createElement("div", { className: "search-wrap" }, [
    createElement("span", { className: "search-icon", text: "⌕", attrs: { "aria-hidden": "true" } }),
    searchInput,
    createElement("span", { className: "search-shortcut", text: "⌘ K", attrs: { "aria-hidden": "true" } }),
  ]));

  const sort = createSelect({
    id: "sort-filter",
    label: "排序方式",
    value: state.sort,
    options: [
      { value: "relevance", label: "相关度优先" },
      { value: "latest", label: "最新发布" },
      { value: "oldest", label: "最早发布" },
      { value: "title", label: "标题排序" },
    ],
    onChange: (sortValue) => updateExplore(state, { sort: sortValue }),
  });
  const toolbar = createElement("div", { className: "results-toolbar" }, [
    createElement("div", { className: "results-meta" }, [
      createElement("button", {
        className: "mobile-filter-button",
        id: "mobile-filter-button",
        type: "button",
        text: "筛选  ☷",
        attrs: { "aria-controls": "filters-panel", "aria-expanded": String(ui.drawerOpen) },
        on: { click: openFilterDrawer },
      }),
      createElement("span", {
        className: "result-count",
        attrs: { "aria-live": "polite", "aria-atomic": "true" },
      }, [
        " 找到 ", createElement("strong", { text: results.length.toLocaleString("zh-CN") }), " 篇",
      ]),
      createElement("span", {
        className: "reading-progress-summary",
        text: `已读 ${currentReadPostCount().toLocaleString("zh-CN")} / ${catalog.stats.postCount}`,
        dataset: { readProgress: "" },
      }),
    ]),
    createElement("div", { className: "sort-wrap" }, sort),
  ]);
  resultsColumn.append(toolbar);

  const chips = createElement("div", { className: "active-filters" });
  state.topics.forEach((topic) => chips.append(makeFilterChip(topic, () => updateExplore(state, { topics: state.topics.filter((item) => item !== topic) }))));
  if (state.tag) chips.append(makeFilterChip(`# ${state.tag}`, () => updateExplore(state, { tag: "" })));
  if (state.category) chips.append(makeFilterChip(state.category, () => updateExplore(state, { category: "" })));
  if (state.year) chips.append(makeFilterChip(state.year, () => updateExplore(state, { year: "" })));
  if (state.series === "series") chips.append(makeFilterChip("仅系列文章", () => updateExplore(state, { series: "" })));
  if (state.series === "standalone") chips.append(makeFilterChip("仅非系列文章", () => updateExplore(state, { series: "" })));
  if (state.read === "read") chips.append(makeFilterChip("只看已读", () => updateExplore(state, { read: "" })));
  if (state.read === "unread") chips.append(makeFilterChip("只看未读", () => updateExplore(state, { read: "" })));
  if (state.level) chips.append(makeFilterChip(LEVEL_LABELS[state.level] ?? state.level, () => updateExplore(state, { level: "" })));
  if (chips.childElementCount) resultsColumn.append(chips);

  if (pageResults.length) {
    const list = createElement("div", { className: "post-list" });
    pageResults.forEach((post) => list.append(makePostCard(post)));
    resultsColumn.append(list);
    const pagination = makePagination({
      current: currentPage,
      total: totalPages,
      href: (page) => hrefFor("/explore", exploreParams({ ...state, page })),
    });
    if (pagination) resultsColumn.append(pagination);
  } else {
    resultsColumn.append(createElement("div", { className: "empty-state" }, [
      createElement("h2", { text: "这个坐标暂时是空的" }),
      createElement("p", { text: "试试减少关键词，或清除一两个筛选条件。" }),
      createElement("a", { className: "button button-ghost", text: "清除筛选", href: "#/explore" }),
    ]));
  }

  layout.append(panel, resultsColumn);
  view.append(layout, makeFooter());
  return view;
}

function renderTopicIndex(catalog) {
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: "Knowledge fields" }),
    createElement("h1", { className: "page-title", text: "主题地图" }),
    createElement("p", {
      className: "page-description",
      text: "同一篇文章可以落在多个主题中。这里呈现的是相互交叠的知识方向，而不是彼此隔绝的分类盒子。",
    }),
    createElement("div", { className: "topic-summary" }, [
      createElement("span", {}, [createElement("strong", { text: catalog.stats.topicCount }), "有效主题"]),
      createElement("span", {}, [createElement("strong", { text: catalog.stats.postCount }), "篇文章"]),
    ]),
  ]));

  const topicMap = new Map(catalog.topics.map((topic) => [topic.name, topic]));
  for (const group of catalog.topicGroups) {
    const section = createElement("section", { className: "content-section compact" });
    section.append(createSectionHeading("Topic group", group.name, ""));
    const grid = createElement("div", { className: "topic-grid" });
    group.topics
      .map((name) => topicMap.get(name))
      .filter((topic) => topic?.count)
      .forEach((topic) => grid.append(makeTopicCard(topic)));
    if (grid.childElementCount) section.append(grid);
    shell.append(section);
  }
  view.append(shell, makeFooter());
  return view;
}

function renderTopicDetail(catalog, name, params) {
  const topic = catalog.topics.find((item) => item.name === name);
  const posts = [...catalog.posts]
    .filter((post) => post.topics.includes(name))
    .sort((a, b) => dateNumber(b.date) - dateNumber(a.date));
  if (!topic && !posts.length) return renderNotFound("没有找到这个主题", "它可能已被合并，或链接中的名称发生了变化。", "#/topics");

  const rawPage = Math.max(1, Number.parseInt(params.get("page") ?? "1", 10) || 1);
  const totalPages = Math.max(1, Math.ceil(posts.length / PAGE_SIZE));
  const page = Math.min(rawPage, totalPages);
  const pagePosts = posts.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const relatedSeries = catalog.series.filter((series) => series.topic === name).sort((a, b) => b.count - a.count);
  const years = unique(posts.map((post) => post.date.slice(0, 4)).filter(Boolean)).sort();

  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: "Topic coordinate" }),
    createElement("h1", { className: "page-title", text: name }),
    createElement("p", {
      className: "page-description",
      text: `这里汇集与「${name}」相关的文章和连续系列，按最新发表时间排列。`,
    }),
    createElement("div", { className: "topic-summary" }, [
      createElement("span", {}, [createElement("strong", { text: posts.length }), "篇文章"]),
      createElement("span", {}, [createElement("strong", { text: relatedSeries.length }), "个系列"]),
      years.length ? createElement("span", {}, [createElement("strong", { text: `${years[0]}—${years.at(-1)}` }), "时间跨度"]) : null,
    ]),
    createElement("div", { className: "inline-actions" }, [
      createElement("a", {
        className: "button button-primary",
        text: "在探索页继续筛选 →",
        href: hrefFor("/explore", (() => { const query = new URLSearchParams(); query.append("topic", name); return query; })()),
      }),
      createElement("a", { className: "button button-ghost", text: "返回主题地图", href: "#/topics" }),
    ]),
  ]));

  if (relatedSeries.length) {
    const section = createElement("section", { className: "content-section compact" });
    section.append(createSectionHeading("Related series", "相关系列", "沿着章节顺序连续阅读。"));
    const grid = createElement("div", { className: "series-grid" });
    relatedSeries.slice(0, 6).forEach((series) => grid.append(makeSeriesCard(series)));
    section.append(grid);
    shell.append(section);
  }

  const articles = createElement("section", {
    className: "content-section compact",
    id: "page-results",
    attrs: { tabindex: "-1" },
  });
  articles.append(createSectionHeading("Articles", `${name}文章`, `第 ${page} / ${totalPages} 页`));
  const list = createElement("div", { className: "post-list" });
  pagePosts.forEach((post) => list.append(makePostCard(post)));
  articles.append(list);
  const pagination = makePagination({
    current: page,
    total: totalPages,
    href: (nextPage) => {
      const query = new URLSearchParams();
      if (nextPage > 1) query.set("page", String(nextPage));
      return hrefFor(`/topics/${encodeURIComponent(name)}`, query);
    },
  });
  if (pagination) articles.append(pagination);
  shell.append(articles);
  view.append(shell, makeFooter());
  return view;
}

function renderSeriesIndex(catalog) {
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: "Connected chapters" }),
    createElement("h1", { className: "page-title", text: "系列目录" }),
    createElement("p", {
      className: "page-description",
      text: "一个概念往往无法在单篇文章里讲完。按章节顺序进入这些长线主题，保留推导与思考的上下文。",
    }),
    createElement("div", { className: "topic-summary" }, [
      createElement("span", {}, [createElement("strong", { text: catalog.series.length }), "个连续系列"]),
      createElement("span", {}, [
        createElement("strong", { text: catalog.series.reduce((sum, series) => sum + series.count, 0) }),
        "篇系列文章",
      ]),
    ]),
  ]));
  const content = createElement("section", { className: "content-section compact" });
  content.append(createSectionHeading("Series atlas", "按篇数浏览", "只展示至少包含两篇文章的系列。"));
  const grid = createElement("div", { className: "series-grid" });
  [...catalog.series].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, "zh-CN"))
    .forEach((series) => grid.append(makeSeriesCard(series)));
  content.append(grid);
  shell.append(content);
  view.append(shell, makeFooter());
  return view;
}

function postsForSeries(catalog, series) {
  const postMap = new Map(catalog.posts.map((post) => [post.id, post]));
  const ordered = series.postIds.map((id) => postMap.get(id)).filter(Boolean);
  if (ordered.length) return ordered;
  const posts = catalog.posts.filter((post) => post.seriesId === series.id || post.series === series.name);
  return posts.sort((a, b) => {
    const aNumbered = a.seriesIndex !== null;
    const bNumbered = b.seriesIndex !== null;
    if (aNumbered !== bNumbered) return aNumbered ? -1 : 1;
    if (aNumbered && a.seriesIndex !== b.seriesIndex) return a.seriesIndex - b.seriesIndex;
    return dateNumber(a.date) - dateNumber(b.date);
  });
}

function renderSeriesDetail(catalog, id) {
  const series = catalog.series.find((item) => item.id === id);
  if (!series) return renderNotFound("没有找到这个系列", "它可能还未达到两篇，或系列规则已经更新。", "#/series");
  const posts = postsForSeries(catalog, series);
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: series.topic || "Long-form series" }),
    createElement("h1", { className: "page-title", text: series.name }),
    createElement("p", {
      className: "page-description",
      text: "章节按系列序号正序排列；没有明确序号时，则沿发表时间向前阅读。",
    }),
    createElement("div", { className: "topic-summary" }, [
      createElement("span", {}, [createElement("strong", { text: posts.length }), "篇文章"]),
      series.startDate && series.endDate
        ? createElement("span", {}, [createElement("strong", { text: `${series.startDate.slice(0, 4)}—${series.endDate.slice(0, 4)}` }), "创作跨度"])
        : null,
      series.topic ? createElement("span", {}, [createElement("strong", { text: series.topic }), "系列主题"]) : null,
    ]),
    createElement("div", { className: "inline-actions" }, [
      series.topic ? createElement("a", { className: "button button-primary", text: "查看主题 →", href: topicHref(series.topic) }) : null,
      createElement("a", { className: "button button-ghost", text: "返回系列目录", href: "#/series" }),
    ]),
  ]));

  const timelineSection = createElement("section", { className: "content-section compact" });
  timelineSection.append(createSectionHeading("Reading path", "章节时间线", "从起点开始，也可以直接进入感兴趣的一章。"));
  const timeline = createElement("div", { className: "timeline" });
  posts.forEach((post, index) => {
    const safeUrl = safeHttpsUrl(post.url);
    const item = createElement("article", {
      className: `timeline-item${isPostRead(post.id) ? " is-read" : ""}`,
    });
    const title = safeUrl
      ? createElement("a", {
          text: post.title,
          href: safeUrl,
          attrs: { target: "_blank", rel: "noopener noreferrer" },
          on: { click: () => updatePostRead(post, true, item, { fromLink: true }) },
        })
      : createElement("span", { text: post.title });
    item.append(
      createElement("span", { className: "timeline-dot", attrs: { "aria-hidden": "true" } }),
      createElement("div", {
        className: "timeline-index",
        text: post.seriesIndex !== null ? `CHAPTER ${String(post.seriesIndex).padStart(2, "0")}` : `ENTRY ${String(index + 1).padStart(2, "0")}`,
      }),
      createElement("h3", {}, title),
      createElement("div", { className: "post-meta" }, [
        createElement("span", { text: formatDate(post.date) }),
        post.level ? createElement("span", { text: LEVEL_LABELS[post.level] ?? post.level }) : null,
        makeReadBadge(post),
      ]),
      post.sourceSummary ? createElement("p", { className: "post-summary", text: post.sourceSummary }) : null,
      createElement("div", { className: "timeline-actions" }, [makeReadToggle(post, item)]),
    );
    timeline.append(item);
  });
  timelineSection.append(timeline);
  shell.append(timelineSection);
  view.append(shell, makeFooter());
  return view;
}

function renderAbout(catalog) {
  const view = createElement("div", { className: "view" });
  const shell = createElement("div", { className: "page-shell" });
  shell.append(createElement("section", { className: "page-hero" }, [
    createElement("p", { className: "eyebrow", text: "About the index" }),
    createElement("h1", { className: "page-title", text: "一个尊重原作的知识入口" }),
    createElement("p", {
      className: "page-description",
      text: "这个项目源于一个朴素的问题：面对持续十余年的高质量写作，怎样让旧文章仍然容易被找到？",
    }),
  ]));
  const content = createElement("section", { className: "content-section compact" });
  const grid = createElement("div", { className: "about-grid" });
  grid.append(
    createElement("article", { className: "about-card" }, [
      createElement("h2", { text: "为什么做这个索引" }),
      createElement("p", { text: "人工整理帖往往难以长期更新。本项目用可重复运行的规则，持续整理标题、主题、系列与少量短小结，让归档保持可搜索、可浏览。" }),
    ]),
    createElement("article", { className: "about-card" }, [
      createElement("h2", { text: "保存什么，不保存什么" }),
      createElement("ul", {}, [
        createElement("li", { text: "保存标题、原文链接、日期、分类、标签、主题与系列信息。" }),
        createElement("li", { text: "仅从明确的小结段落提取有限长度的短摘录。" }),
        createElement("li", { text: "不镜像、不复制，也不保存任何文章正文。" }),
        createElement("li", { text: "已读状态只保存在当前浏览器，不上传到服务器。" }),
        createElement("li", { text: "访问量使用 Cloudflare Web Analytics 匿名汇总，不使用 Cookie，也不用于识别个人。" }),
      ]),
    ]),
    createElement("article", { className: "about-card" }, [
      createElement("h2", { text: "如何持续更新" }),
      createElement("p", { text: "自动任务每天检查归档页，只为新增文章补充元数据，并在本地规则下重新分类和生成页面。访问文章页时串行请求并保持间隔，尽量减少对原站的影响。" }),
    ]),
    createElement("article", { className: "about-card" }, [
      createElement("h2", { text: "版权与归属" }),
      createElement("p", { text: "本站是社区维护的非官方导航。文章内容、标题及相关权利归原作者与科学空间所有；阅读、引用与讨论请始终前往原文。" }),
      createElement("div", { className: "inline-actions" }, [
        externalLink("访问科学空间 ↗", "https://spaces.ac.cn/", "button button-primary"),
        externalLink("查看项目源码 ↗", "https://github.com/caojiaolong/spaces-index", "button button-ghost"),
      ]),
    ]),
  );
  content.append(grid);
  shell.append(content);
  view.append(shell, makeFooter());
  return view;
}

function renderNotFound(title = "没有找到这个坐标", copy = "检查链接，或回到探索页继续浏览。", href = "#/explore") {
  const view = createElement("div", { className: "view" });
  view.append(createElement("section", { className: "error-state" }, [
    createElement("p", { className: "eyebrow", text: "404 / undefined" }),
    createElement("h1", { text: title }),
    createElement("p", { text: copy }),
    createElement("a", { className: "button button-primary", text: "返回索引", href }),
  ]), makeFooter());
  return view;
}

function routeTitle(parts) {
  if (!parts.length) return "科学空间索引";
  if (parts[0] === "explore") return "探索文章 · 科学空间索引";
  if (parts[0] === "topics") return parts[1] ? `${parts[1]} · 科学空间索引` : "主题地图 · 科学空间索引";
  if (parts[0] === "series") return parts[1] ? "系列阅读 · 科学空间索引" : "系列目录 · 科学空间索引";
  if (parts[0] === "about") return "关于 · 科学空间索引";
  return "未找到 · 科学空间索引";
}

function updateNavigation(parts) {
  const root = parts[0] ?? "home";
  const current = ["explore", "topics", "series", "about"].includes(root) ? root : "home";
  document.querySelectorAll("[data-nav]").forEach((link) => {
    if (link.dataset.nav === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function syncAfterRender(parts) {
  const isExplore = parts[0] === "explore";
  if (!isExplore) closeFilterDrawer({ restoreFocus: false });
  else if (ui.drawerOpen && innerWidth <= 760) {
    document.body.classList.add("drawer-open");
    drawerBackdrop.hidden = false;
    document.querySelector("#filters-panel")?.classList.add("is-open");
  } else {
    document.body.classList.remove("drawer-open");
    drawerBackdrop.hidden = true;
    document.querySelector("#filters-panel")?.classList.remove("is-open");
  }

  if (ui.focusAfterRender) {
    const id = ui.focusAfterRender;
    ui.focusAfterRender = null;
    requestAnimationFrame(() => {
      const target = document.getElementById(id);
      if (!target) return;
      target.focus({ preventScroll: true });
      if (typeof target.setSelectionRange === "function") {
        const length = target.value.length;
        target.setSelectionRange(length, length);
      }
    });
  } else if (ui.reopenSelect) {
    const id = ui.reopenSelect;
    ui.reopenSelect = null;
    requestAnimationFrame(() => document.querySelector(`[data-select-id="${CSS.escape(id)}"]`)?._openSelect());
  }
}

function renderRoute() {
  if (!ui.catalog) return;
  const { path, parts, params } = parseLocation();
  const pathChanged = ui.lastPath !== null && ui.lastPath !== path;
  let view;
  if (!parts.length) view = renderHome(ui.catalog);
  else if (parts[0] === "explore" && parts.length === 1) view = renderExplore(ui.catalog, params);
  else if (parts[0] === "topics" && parts.length === 1) view = renderTopicIndex(ui.catalog);
  else if (parts[0] === "topics" && parts.length === 2) view = renderTopicDetail(ui.catalog, parts[1], params);
  else if (parts[0] === "series" && parts.length === 1) view = renderSeriesIndex(ui.catalog);
  else if (parts[0] === "series" && parts.length === 2) view = renderSeriesDetail(ui.catalog, parts[1]);
  else if (parts[0] === "about" && parts.length === 1) view = renderAbout(ui.catalog);
  else view = renderNotFound();

  main.replaceChildren(view);
  document.title = routeTitle(parts);
  updateNavigation(parts);
  syncAfterRender(parts);
  ui.lastPath = path;
  if (ui.scrollAfterRender) {
    ui.scrollAfterRender = false;
    requestAnimationFrame(() => {
      const target = document.querySelector("#page-results");
      if (!target) return;
      const top = target.getBoundingClientRect().top + scrollY - header.offsetHeight - 16;
      scrollTo({ top: Math.max(0, top), behavior: reduceMotion.matches ? "auto" : "smooth" });
      target.focus({ preventScroll: true });
    });
  } else if (pathChanged) {
    scrollTo({ top: 0, behavior: "auto" });
    main.focus({ preventScroll: true });
  }
  if (!parts.length) requestAnimationFrame(animateHomeStats);
  scheduleScrollState();
}

function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return ["system", "light", "dark"].includes(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

function applyTheme(preference) {
  const resolved = preference === "system" ? (colorScheme.matches ? "dark" : "light") : preference;
  document.documentElement.dataset.theme = resolved;
  document.documentElement.dataset.themePreference = preference;
  const labels = { system: "跟随系统", light: "浅色", dark: "深色" };
  const icons = { system: "◐", light: "☼", dark: "☾" };
  themeLabel.textContent = labels[preference];
  themeIcon.textContent = icons[preference];
  themeButton.setAttribute("aria-label", `颜色主题：${labels[preference]}。点击切换`);
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", resolved === "dark" ? "#090c14" : "#f5f6fa");
}

function cycleTheme() {
  const current = document.documentElement.dataset.themePreference || readStoredTheme();
  const order = ["system", "light", "dark"];
  const next = order[(order.indexOf(current) + 1) % order.length];
  try { localStorage.setItem(THEME_KEY, next); } catch { /* Storage may be disabled. */ }
  applyTheme(next);
}

let scrollStateFrame = 0;

function syncScrollState() {
  scrollStateFrame = 0;
  const maximum = Math.max(1, document.documentElement.scrollHeight - innerHeight);
  const progress = Math.min(1, Math.max(0, scrollY / maximum));
  const visible = scrollY > Math.max(320, innerHeight * .55);
  header.classList.toggle("is-scrolled", scrollY > 8);
  backToTop.style.setProperty("--scroll-progress", `${progress}turn`);
  backToTop.classList.toggle("is-visible", visible);
  backToTop.setAttribute("aria-hidden", String(!visible));
  backToTop.tabIndex = visible ? 0 : -1;
}

function scheduleScrollState() {
  if (scrollStateFrame) return;
  scrollStateFrame = requestAnimationFrame(syncScrollState);
}

themeButton.addEventListener("click", cycleTheme);
skipLink.addEventListener("click", (event) => {
  event.preventDefault();
  main.focus();
});
colorScheme.addEventListener("change", () => {
  if ((document.documentElement.dataset.themePreference || readStoredTheme()) === "system") applyTheme("system");
});
backToTop.addEventListener("click", () => {
  scrollTo({ top: 0, behavior: reduceMotion.matches ? "auto" : "smooth" });
  setTimeout(() => main.focus({ preventScroll: true }), reduceMotion.matches ? 0 : 360);
});
window.addEventListener("hashchange", renderRoute);
window.addEventListener("scroll", scheduleScrollState, { passive: true });
window.addEventListener("resize", () => {
  const panel = document.querySelector("#filters-panel");
  if (innerWidth > 760) {
    if (ui.drawerOpen) closeFilterDrawer({ restoreFocus: false });
    panel?.removeAttribute("inert");
    panel?.setAttribute("aria-hidden", "false");
    panel?.setAttribute("role", "region");
    panel?.removeAttribute("aria-modal");
  } else if (!ui.drawerOpen) {
    panel?.setAttribute("inert", "");
    panel?.setAttribute("aria-hidden", "true");
    panel?.setAttribute("role", "dialog");
    panel?.setAttribute("aria-modal", "true");
  }
  scheduleScrollState();
}, { passive: true });
drawerBackdrop.addEventListener("click", () => closeFilterDrawer());

document.addEventListener("keydown", (event) => {
  if (event.key === "Tab" && ui.drawerOpen) {
    const panel = document.querySelector("#filters-panel");
    const focusable = panel ? [...panel.querySelectorAll("button:not([disabled]), input:not([disabled]), a[href]")]
      .filter((element) => element.getClientRects().length) : [];
    if (focusable.length) {
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
    event.preventDefault();
    if (parseLocation().parts[0] === "explore") document.querySelector("#explore-search")?.focus();
    else {
      ui.focusAfterRender = "explore-search";
      location.hash = "#/explore";
    }
  }
  if (event.key === "Escape" && ui.drawerOpen) closeFilterDrawer();
});

function renderLocalPreviewHelp() {
  main.replaceChildren(createElement("section", { className: "error-state local-preview-help" }, [
    createElement("p", { className: "eyebrow", text: "Local preview" }),
    createElement("h1", { text: "请通过本地 HTTP 服务预览" }),
    createElement("p", {
      text: "直接双击 HTML 会受到浏览器本地文件安全策略限制，页面无法读取 catalog.json。请在仓库目录运行：",
    }),
    createElement("code", {
      className: "preview-command",
      text: "uv run python -m http.server 8000 --directory _site",
    }),
    createElement("p", { text: "然后在浏览器访问 http://127.0.0.1:8000/" }),
  ]));
}

async function start() {
  applyTheme(readStoredTheme());
  void loadGitHubStarCount();
  ui.readPostIds = loadReadPostIds();
  if (location.protocol === "file:") {
    renderLocalPreviewHelp();
    return;
  }
  try {
    const response = await fetch("./catalog.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const raw = await response.json();
    ui.catalog = normalizeCatalog(raw);
    const validPostIds = new Set(ui.catalog.posts.map((post) => post.id));
    const storedReadCount = ui.readPostIds.size;
    ui.readPostIds = new Set([...ui.readPostIds].filter((postId) => validPostIds.has(postId)));
    if (ui.readPostIds.size !== storedReadCount) persistReadPostIds();
    if (!location.hash) history.replaceState(null, "", `${location.pathname}${location.search}#/`);
    renderRoute();
  } catch (error) {
    console.error("Unable to load catalog", error);
    main.replaceChildren(createElement("section", { className: "error-state" }, [
      createElement("p", { className: "eyebrow", text: "Catalog unavailable" }),
      createElement("h1", { text: "知识坐标暂时无法展开" }),
      createElement("p", { text: "catalog.json 加载失败。请稍后刷新，或前往 GitHub 查看静态元数据。" }),
      externalLink("在 GitHub 查看索引 ↗", "https://github.com/caojiaolong/spaces-index", "button button-primary"),
    ]));
  }
}

start();
scheduleScrollState();
