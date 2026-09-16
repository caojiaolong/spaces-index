const { test } = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

// Stub browser startup only. Load the actual shipped scripts and keep fetch
// unresolved so neither startup nor these pure catalogue tests access a network.
function runtime(stored = {}) {
  const element = { addEventListener() {}, setAttribute() {}, dataset: {}, style: { setProperty() {} } };
  const storage = new Map(Object.entries(stored));
  const context = vm.createContext({
    document: { querySelector: selector => ["#continue-reading", "#reading-percent[data-reading-post]"].includes(selector) ? null : element,
      querySelectorAll: () => [], documentElement: element, addEventListener() {} },
    window: { addEventListener() {} },
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    localStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    location: { protocol: "http:" },
    history: { scrollRestoration: "auto" },
    fetch: () => new Promise(() => {}),
    setTimeout: () => 0, clearTimeout() {}, requestAnimationFrame: () => 0,
    URL, URLSearchParams, AbortController, console,
  });
  for (const file of ["library.js", "reader.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "web", file), "utf8"), context);
  }
  return (expression) => JSON.parse(vm.runInContext(`JSON.stringify(${expression})`, context) ?? "null");
}

test("reading percentage is persistent, monotonic and only complete at 100", () => {
  const key = 'spaces-index-reading-progress-v1';
  const run = runtime({ [key]: JSON.stringify({ 1: 42, 2: 99.9, 3: 100, 4: 200, 5: '100' }) });
  assert.deepEqual(run(`[getPostProgress('1'), getPostProgress('2'), isPostRead('2'), isPostRead('3'), getPostProgress('4'), getPostProgress('5')]`), [42,99,false,true,0,0]);
  run(`recordReadingProgress('1', 63)`);
  run(`recordReadingProgress('1', 12)`);
  assert.equal(run(`getPostProgress('1')`), 63);
  // A stale tab writing another article must preserve a newer stored maximum.
  run(`localStorage.setItem(READING_PROGRESS_KEY, JSON.stringify({'1':82, '3':100}))`);
  run(`recordReadingProgress('2', 100)`);
  assert.deepEqual(run(`[getPostProgress('1'), [...ui.readPostIds].sort()]`), [82,['2','3']]);
  const saved = run(`localStorage.getItem(READING_PROGRESS_KEY)`);
  assert.equal(runtime({ [key]: saved })(`getPostProgress('1')`), 82);
  assert.equal(run(`isPostRead('1')`), false);
});

test("legacy link clicks do not become fabricated 100 percent progress", () => {
  const run = runtime({ 'spaces-index-read-posts-v1': '["1","2"]' });
  assert.deepEqual(run(`[getPostProgress('1'), isPostRead('1'), currentReadPostCount()]`), [0,false,0]);
  assert.equal(run(`localStorage.getItem('spaces-index-read-posts-v1')`), '["1","2"]');
  assert.deepEqual(runtime({ 'spaces-index-reading-progress-v1': 'invalid' })(`[...ui.readingProgress]`), []);
});

test("body progress waits for the last line and accounts for fixed navigation", () => {
  const run = runtime();
  const percent = obj => run(`window.SpacesReader.readingPercent(${JSON.stringify(obj)})`);
  assert.equal(percent({top:500,bottom:4500,scroll:0,viewport:1000,insetTop:76}), 0);
  assert.equal(percent({top:-2400,bottom:1600,scroll:2900,viewport:1000,insetTop:76}), 80);
  assert.equal(percent({top:-2990,bottom:1010,scroll:3490,viewport:1000,insetTop:76}), 99);
  assert.equal(percent({top:-3000,bottom:1000,scroll:3500,viewport:1000,insetTop:76}), 100);
  assert.equal(percent({top:-3000,bottom:1000,scroll:3500,viewport:1000,insetTop:76,insetBottom:60}), 98);
  assert.equal(percent({top:500,bottom:800,scroll:0,viewport:1000,insetTop:76}), 100);
  assert.equal(percent({top:0,bottom:0,scroll:0,viewport:1000}), 0);
});

test("topic evidence counts distinct shared articles and normalized shared tags", () => {
  const run = runtime();
  const result = run(`topicConnections(normalizeCatalog({posts: [
    {id: '1', topics: ['A', 'B'], sourceTags: ['ＡＩ', 'AI', 'shared']},
    {id: '2', topics: ['A'], sourceTags: ['ai']},
    {id: '3', topics: ['B'], sourceTags: ['ai', 'b-only']},
    {id: '4', topics: ['C'], sourceTags: ['AI']},
    {id: '5', topics: ['D'], sourceTags: ['unrelated']}
  ]}), 'A')`);
  assert.deepEqual(result.map(({ topic, postCount, tags }) => [topic.name, postCount, tags.length]), [["B", 1, 2], ["C", 0, 1]]);
  assert.equal(result[0].tags[0].toLowerCase(), "ai");
  assert.deepEqual(run(`topicConnections(normalizeCatalog({posts: []}), 'missing')`), []);
});

test("intersection links filter both topics while ordinary multi-select remains a union", () => {
  const run = runtime();
  const posts = `{posts: [
    {id:'1', title:'Both', topics:['A','B']},
    {id:'2', title:'Only A', topics:['A']},
    {id:'3', title:'Only B', topics:['B']},
    {id:'4', title:'Neither', topics:['C']}
  ]}`;
  assert.deepEqual(run(`searchAndFilter(normalizeCatalog(${posts}), readExploreState(new URLSearchParams('topic=A&with=B'))).map(p=>p.id)`), ["1"]);
  assert.deepEqual(run(`searchAndFilter(normalizeCatalog(${posts}), readExploreState(new URLSearchParams('topic=A&topic=B'))).map(p=>p.id)`), ["1", "2", "3"]);
  assert.equal(run(`exploreParams(readExploreState(new URLSearchParams('topic=A&with=B'))).get('with')`), "B");
});

test("continue reading follows series order, including gaps and fully read series", () => {
  const run = runtime();
  const catalog = `{posts: [{id:'2'}, {id:'1'}, {id:'3'}]}`;
  const series = `{postIds:['1','2','3'], id:'series-a', name:'A'}`;
  assert.equal(run(`seriesReadingState(${catalog}, ${series}, new Set(['2'])).next.id`), "1");
  assert.deepEqual(run(`seriesReadingState(${catalog}, ${series}, new Set(['1', '2']))`), { total: 3, read: 2, next: { id: "3" } });
  assert.deepEqual(run(`seriesReadingState(${catalog}, ${series}, new Set(['1','2','3','stale-id']))`), { total: 3, read: 3, next: null });
  assert.equal(run(`seriesChapterHref(${series}, {id:'3'} )`), "#/series/series-a?chapter=3");
});

test("series cover identity survives reloads, reordering and metadata updates", () => {
  const run = runtime();
  const original = run(`seriesAppearance({id:'series-数学-42', name:'数学笔记', topic:'数学', count:3})`);
  assert.deepEqual(runtime()(`seriesAppearance({id:'series-数学-42', name:'数学笔记', topic:'数学', count:3})`), original);
  run(`Array.from({length:20}, (_, i) => seriesAppearance({id:String(i)}))`);
  assert.deepEqual(run(`seriesAppearance({id:'series-数学-42', name:'更新的标题', topic:'几何', count:30})`), original);
  assert.deepEqual(run(`seriesAppearance({name:'没有 ID 的系列'})`), runtime()(`seriesAppearance({name:'没有 ID 的系列'})`));
});

test("series in the same topic get varied artwork, colours and geometric seeds", () => {
  const run = runtime();
  const appearances = run(`Array.from({length:300}, (_, i) => seriesAppearance({id:'series-' + i, topic:'数学'}))`);
  const kinds = new Set(appearances.map((item) => item.kind));
  assert.equal(kinds.size, 14);
  assert.ok(new Set(appearances.map((item) => item.hue)).size > 120);
  assert.equal(new Set(appearances.map((item) => item.seed)).size, appearances.length);
  for (const kind of kinds) {
    assert.ok(new Set(appearances.filter((item) => item.kind === kind).map((item) => item.hue)).size > 4);
  }
  for (const item of appearances) {
    assert.ok(Number.isInteger(item.hue) && item.hue >= 0 && item.hue < 360);
    assert.ok(item.saturation >= 28 && item.saturation <= 42);
  }
});
