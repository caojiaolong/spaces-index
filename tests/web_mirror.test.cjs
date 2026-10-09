const { test } = require("node:test");
const assert = require("node:assert/strict");
const { publicMarkdownUrl, questionPrompt, aiProviders, aiProvider, aiUrl, preferredAi, setPreferredAi } = require("../web/mirror.js");

test("AI preference defaults to GPT and remains usable when browser storage is disabled", () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  const values = new Map();
  try {
    Object.defineProperty(globalThis, "localStorage", { configurable: true, value: {
      getItem: key => values.get(key), setItem: (key, value) => values.set(key, value)
    } });
    assert.equal(preferredAi().id, "chatgpt");
    setPreferredAi("gemini");
    assert.equal(preferredAi().id, "gemini");
    setPreferredAi("javascript:alert(1)");
    assert.equal(preferredAi().id, "chatgpt");
    values.set("spaces-reader-ai", "obsolete-provider");
    assert.equal(preferredAi().id, "chatgpt");
    Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw Error("denied"); } });
    assert.equal(preferredAi().id, "chatgpt");
    assert.equal(setPreferredAi("deepseek").id, "deepseek");
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else delete globalThis.localStorage;
  }
});

test("provider handoffs preserve complete queries and offer copying for unsupported URLs", () => {
  const prompt = String.raw`请解释这一段：$\alpha+\beta$ & 中文 # + 末尾`;
  for (const provider of aiProviders) {
    const url = aiUrl(prompt, provider.id);
    assert.equal(new URL(provider.url).protocol, "https:");
    if (provider.query) {
      assert.equal(new URL(url).searchParams.get("q"), prompt);
      assert.equal(new URL(url).host, new URL(provider.url).host);
      assert.equal(aiUrl("中".repeat(700), provider.id), null);
    } else assert.equal(url, null);
  }
  assert.equal(new URL(aiUrl(prompt, "perplexity")).searchParams.get("s"), "o");
  assert.equal(aiProvider("unknown").id, "chatgpt");
});

test("public Markdown URLs preserve deployment paths and exclude local preview addresses", () => {
  const expected = "https://caojiaolong.github.io/spaces-index/mirror/9119/article.md";
  for (const baseUrl of ["http://127.0.0.1:8765/#/article/9119", "https://localhost/app/index.html", "https://192.168.1.2/", "https://[::1]/", "file:///D:/site/index.html", "invalid", "https://user:pass@example.com/app/"]) {
    assert.equal(publicMarkdownUrl("9119", { baseUrl }), expected);
  }
  assert.equal(publicMarkdownUrl("9119", { baseUrl: "https://example.com/app/index.html?q=x#/article/9119" }), "https://example.com/app/mirror/9119/article.md");
  assert.equal(publicMarkdownUrl("9119", { baseUrl: "https://preview.example.com/", localPreview: true }), expected);
  assert.throws(() => publicMarkdownUrl("../secrets"));
  assert.throws(() => questionPrompt({ id: "../secrets", title: "测试", markdownUrl: expected }));
});

test("Chinese, special characters and original LaTeX survive the ChatGPT URL exactly", () => {
  const selection = String.raw`选段 & + #：$\alpha_t > 0$ 和 \begin{aligned}x&=1\\y&=2\end{aligned}`;
  const prompt = questionPrompt({ id: "9119", title: "测试文章", markdownUrl: publicMarkdownUrl("9119"), selection });
  assert.equal(new URL(aiUrl(prompt)).searchParams.get("q"), prompt);
  assert.ok(prompt.includes(selection));
  assert.ok(prompt.includes("无法读取时明确说明"));
  assert.ok(prompt.includes("联网工具获取上面的完整 Markdown 正文"));
  assert.ok(prompt.includes("https://raw.githubusercontent.com/caojiaolong/spaces-index/main/data/articles/9119/article.md"));
  assert.ok(!prompt.includes("article.txt"));
  assert.ok(prompt.endsWith("请解释这一段文字。"));
  assert.ok(prompt.includes("不冒充作者"));
  assert.ok(!prompt.includes("#/article"));
});

test("handoff budget uses encoded URL length and never truncates long content", () => {
  const overhead = new URL("https://chatgpt.com/?q=").href.length;
  const limit = "a".repeat(6000 - overhead);
  assert.equal(aiUrl(limit).length, 6000);
  assert.equal(aiUrl(limit + "a"), null);
  assert.equal(aiUrl("中".repeat(700)), null);
  const selection = "长选段".repeat(1000) + "这是末尾";
  const prompt = questionPrompt({ id: "9119", title: "测试", markdownUrl: publicMarkdownUrl("9119"), selection });
  assert.equal(aiUrl(prompt), null);
  assert.ok(prompt.includes(selection));
});

test("full fallback keeps Markdown byte-for-byte and partial formulas retain source context", () => {
  const markdown = "# 标题\n\n> 苏剑林 · CC BY-NC-ND 2.5 CN\n\n$$\\alpha+\\beta$$\n末尾\n";
  const prompt = questionPrompt({ id: "9119", title: "测试", markdownUrl: publicMarkdownUrl("9119"), markdown });
  assert.ok(prompt.endsWith(markdown));
  assert.ok(!prompt.includes("联网工具获取"));
  assert.ok(!prompt.includes("raw.githubusercontent.com"));
  const latex = String.raw`$\alpha_t+\beta_t$`;
  const partial = questionPrompt({ id: "11882", title: "测试", markdownUrl: publicMarkdownUrl("11882"), selection: "α", partialFormulas: [{ selected: "α", latex }] });
  assert.ok(partial.includes("https://raw.githubusercontent.com/caojiaolong/spaces-index/main/data/articles/11882/article.md"));
  assert.ok(partial.includes(latex));
  assert.equal(partial.split(latex).length - 1, 1);
});
