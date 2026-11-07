# Project website

The static website lives in `docs/`. Serve this directory to preview it:

```sh
python3 -m http.server 8765 --directory docs
```

The homepage is `index.html`, the method results are `openmark.html`, the developer guide is `docs.html`, and the existing GPU run interface is `app.html`. Recorded demo assets remain in `static/runs/`; this migration does not change benchmark data or demo logic.

The public URL is https://loopleap-lab.github.io/OpenMarkX/. GitHub Pages can deploy `main:/docs` if this repository's plan and visibility permit it. Otherwise, publish the four HTML pages, `.nojekyll`, and `static/` to the `OpenMarkX/` directory of the lab's public website repository. Never copy the private code repository or its history to the public website repository.
