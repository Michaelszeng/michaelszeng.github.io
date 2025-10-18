# Michael Zeng's Website

Personal website integrating HTML with Obsidian-style Latex subpages.

## Build & Deploy

After modifying markdown files in `content/`:

```bash
# Build the site (generates HTML in public/)
node quartz/bootstrap-cli.mjs build

# If changes don't appear, clear cache and rebuild:
rmdir /S /Q .quartz-cache 2>nul & node quartz/bootstrap-cli.mjs build

# Preview: open public/diffusion-policy-loss.html in browser (Ctrl+Shift+R to hard refresh)
# OR run dev server (better for development):
node quartz/bootstrap-cli.mjs build --serve  # Visit http://localhost:8080

# Deploy
git add .
git commit -m "Update content"
git push
```

## Structure

- `index.html` - Custom portfolio homepage
- `content/` - Markdown source files (with LaTeX support)
- `public/` - Generated HTML (linked from index.html)
- `quartz.config.ts` - Site configuration
- `quartz.layout.ts` - Page layout/components

---

*Template inspired by [Andy Zeng](https://andyzeng.github.io/) and [Jon Barron](https://jonbarron.info/)*
