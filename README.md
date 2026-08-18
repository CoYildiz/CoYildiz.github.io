# CoYildiz

Personal blog, built with [Hugo](https://gohugo.io) using [hugo-theme-reimu](https://github.com/D-Sketon/hugo-theme-reimu). Bilingual: English (default) and Turkish (`/tr/`).

## Local development

Requires [Hugo](https://gohugo.io/installation/) (extended, >= 0.158.0).

```sh
git submodule update --init --recursive
hugo server
```

Site is served at `http://localhost:1313/`.

## Adding a post

Create a new Markdown file under `content/post/`:

- English posts: `content/post/my-post.md`
- Turkish posts: `content/post/my-post.tr.md`

Front matter fields: `title`, `date`, `lastmod`, `description`, `tags`, `categories`, `draft`.

## Structure

- `content/` — posts and pages (`about.md`/`about.tr.md`, `archives/`, `post/`, `all.md`/`all.tr.md` for the combined EN+TR view)
- `config/_default/params.yml` — theme configuration
- `i18n/` — UI string overrides (Turkish translations + shared additions)
- `layouts/` — project-level template overrides (theme customizations: home page, header/language switcher, accent color picker)
- `data/covers.yml` — random post cover images (empty, falls back to the site banner)
- `static/` — banner and avatar images
- `themes/reimu` — the theme, as a git submodule

## Deployment

Pushing to `main` builds and deploys to GitHub Pages via `.github/workflows/deploy.yml`.
