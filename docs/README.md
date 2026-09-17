# 文档导航

[项目首页](../README.md) · [在线阅读](https://caojiaolong.github.io/spaces-index/)

查文章从主题索引开始；运行和维护项目从下方指南开始。这里的命令均在仓库根目录执行。

## 文章索引

[主题索引与统计](topics/README.md)按主题整理文章与系列，保留日期、原站分类、标签、系列序号、小结摘录和备注。在线阅读也可以使用[文章检索](https://caojiaolong.github.io/spaces-index/#/explore)或[系列书架](https://caojiaolong.github.io/spaces-index/#/series)。

## 使用与维护

| 想做什么 | 文档 |
| --- | --- |
| 本地运行、阅读、复制公式、预览图片 | [本地预览与阅读](guides/local-preview.md) |
| 更新正文、同步索引、缓存图片、构建与部署 | [日常维护](guides/maintenance.md) |
| 查找提取失败原因、离线修复、复查指定文章 | [正文提取与失败恢复](guides/local-extraction.md) |
| 了解正文校验、撤回、许可与已知边界 | [原文 Markdown：校验与许可](guides/mirror.md) |

## 文档如何更新

```text
docs/
├── README.md       文档导航
├── topics/         自动生成的主题索引与详细元数据
│   └── README.md   主题统计与各主题入口
└── guides/         人工维护的使用、维护与许可文档
```

根目录 `README.md` 和 `topics/` 由 `scripts/render_markdown.py` 生成；日常统一更新会自动运行它。已有分类数据时，也可仅重新生成文档：

```bash
uv run python scripts/render_markdown.py
```

主题、标签或系列有误时，修改分类规则或 [`data/overrides.yaml`](../data/overrides.yaml)，再运行离线更新。不要直接修改生成页，否则下次更新会覆盖。根 README 的介绍文字在生成脚本中维护；本文和 `guides/` 可直接编辑，生成脚本不会覆盖。

文章正文、快照、校验报告和图片统一保存在 [`data/articles/`](../data/articles/)，数据布局见 [data 说明](../data/README.md)。
