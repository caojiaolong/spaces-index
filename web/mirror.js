/* Shared integrity and clipboard operations. No article HTML is injected. */
(function (root) {
  "use strict";
  function path(id, extension = "md") {
    if (!/^[1-9]\d*$/.test(String(id)) || !["md", "json"].includes(extension)) throw new Error("文章编号无效");
    return `./mirror/${id}/article.${extension}`;
  }
  async function verify(text, expected) {
    if (!/^[a-f0-9]{64}$/.test(expected || "") || !root.crypto?.subtle) throw new Error("无法校验文件，请通过 HTTPS 或 localhost 打开页面。");
    const bytes = await root.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    const actual = Array.from(new Uint8Array(bytes), n => n.toString(16).padStart(2, "0")).join("");
    if (actual !== expected) throw new Error("文件校验失败，请刷新后重试或阅读原文。");
    return text;
  }
  async function load(id, expected) {
    const response = await fetch(path(id), { cache: "no-store" });
    if (!response.ok) throw new Error("Markdown 暂不可用，可能已下架，请阅读原文。");
    return verify(await response.text(), expected);
  }
  async function copy(text) {
    if (!root.navigator.clipboard?.writeText) throw new Error("浏览器不允许复制，请在 Markdown 源码视图中手动全选复制。");
    await root.navigator.clipboard.writeText(text);
  }
  root.SpacesMirror = { path, verify, load, copy };
  if (typeof module !== "undefined") module.exports = root.SpacesMirror;
})(typeof window === "undefined" ? globalThis : window);
