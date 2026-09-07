const { test } = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

// Stub browser startup only. Load the actual shipped scripts and keep fetch
// unresolved so neither startup nor these pure catalogue tests access a network.
function runtime() {
  const element = { addEventListener() {}, setAttribute() {}, dataset: {}, style: { setProperty() {} } };
  const context = vm.createContext({
    document: { querySelector: () => element, documentElement: element, addEventListener() {} },
    window: { addEventListener() {} },
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    localStorage: { getItem: () => null, setItem() {} },
    location: { protocol: "http:" },
    fetch: () => new Promise(() => {}),
    setTimeout: () => 0, clearTimeout() {}, requestAnimationFrame: () => 0,
    URL, URLSearchParams, AbortController, console,
  });
  for (const file of ["library.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "web", file), "utf8"), context);
  }
  return (expression) => JSON.parse(vm.runInContext(`JSON.stringify(${expression})`, context));
}

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
