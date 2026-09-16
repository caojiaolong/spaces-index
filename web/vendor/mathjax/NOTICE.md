# MathJax 4.1.3

The reader uses CommonHTML with actual text nodes so readers can select and copy
individual symbols. The `mathjax-tex` font preserves the previous TeX appearance.

Vendored unmodified files:

- `mathjax@4.1.3`: `tex-chtml-nofont.js`, explicit TeX extensions,
  `a11y/assistive-mml.js` and Apache-2.0 `LICENSE`.
- `@mathjax/mathjax-tex-font@4.1.3`: `chtml.js`, `chtml/woff2/`, under `tex-font/`.
- `@mathjax/mathjax-mhchem-font-extension@4.1.3`: `chtml.js`, `chtml/woff2/`,
  under `extensions/mathjax-mhchem-font-extension/`.

JavaScript packages use Apache-2.0. Fonts retain their embedded copyright and
license notices: TeX fonts use SIL OFL 1.1 (see `FONT-LICENSE.txt`); the mhchem
fonts carry the GUST Font License (see the extension's `FONT-LICENSE.txt`).

All renderer, extension and font URLs resolve locally. Article-triggered
`autoload` and `require` are disabled. The original `\require{AMScd}` is retained
in Markdown; its functionality comes from the explicitly loaded `amscd`
extension. No article LaTeX is rewritten.

- Source: https://github.com/mathjax/MathJax/tree/4.1.3
- Documentation: https://docs.mathjax.org/en/latest/
- Character selection: https://github.com/mathjax/MathJax/issues/2240

Package integrity (SHA-512, verified before extracting):

- `mathjax@4.1.3`
  - Package: https://registry.npmjs.org/mathjax/-/mathjax-4.1.3.tgz
  - `sha512-BN/8Pkgn7G1pIDYJqd9md+JHsE/jydSYbyOZnSdSA0WziuVO8mRxdYiWFumkVVly/8U+hm9DpIIoWuvySverzw==`
- `@mathjax/mathjax-tex-font@4.1.3`
  - Package: https://registry.npmjs.org/@mathjax/mathjax-tex-font/-/mathjax-tex-font-4.1.3.tgz
  - `sha512-9B78brBEmmAwyumaREIyM1gF2HgDamDIXA36UyeXWxX7XrpLdmoSB5NLIf/c6tEA4L6xNrIdMxfgY6YX/kcSNw==`
- `@mathjax/mathjax-mhchem-font-extension@4.1.3`
  - Package: https://registry.npmjs.org/@mathjax/mathjax-mhchem-font-extension/-/mathjax-mhchem-font-extension-4.1.3.tgz
  - `sha512-WFx0IooitEJq1TXc9V941o8eaZdICSPn8FpsIBDdhkMhb0if61jCvk6vtQN84dlLnNEorK011NfpirMkHYygSQ==`
