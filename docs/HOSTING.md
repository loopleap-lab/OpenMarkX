# Project website

The static website lives in `docs/`. Serve this directory to preview it:

```sh
python3 -m http.server 8765 --directory docs
```

The homepage is `index.html`, the method results are `openmark.html`, the developer guide is `docs.html`, and the existing GPU run interface is `app.html`. Recorded demo assets remain in `static/runs/`; this migration does not change benchmark data or demo logic.

The public website is https://loopleap-lab.github.io/OpenMarkX/ and its interactive demos are at [the demo URL](https://loopleap-lab.github.io/OpenMarkX/demo/). This entry point opens the existing `#demos` section on the homepage; the implementation and recorded data are shared.

GitHub Pages publishes this repository's `main` branch from `/docs`. Keep `.nojekyll` in that directory. Merged changes to the HTML and `static/` assets deploy automatically; no Claude account or separate website repository is required.

The site displays the recorded experiments interactively. The optional GPU run interface connects to a server started by the user with `openmarkx serve`; GitHub Pages itself does not run model training or inference.
