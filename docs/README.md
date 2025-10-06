# Driftlock Choir GitHub Pages Website

This directory contains the Jekyll-based GitHub Pages website for the Driftlock Choir project.

## Local Development

1. Install Ruby and Jekyll:
   ```bash
   gem install bundler jekyll
   ```

2. Install dependencies:
   ```bash
   cd docs
   bundle install
   ```

3. Serve locally:
   ```bash
   bundle exec jekyll serve
   ```

4. Open http://localhost:4000 in your browser

## Structure

- `_config.yml` - Jekyll configuration
- `_layouts/` - Page layouts
- `assets/` - CSS, JavaScript, images, and audio files
- `*.html` / `*.md` - Page content files
- `Gemfile` - Ruby dependencies

## Features

- Interactive beat-note visualizations
- Audio demonstrations with Web Audio API
- Real-time parameter manipulation
- Mathematical equation rendering with MathJax
- Responsive design for all devices
- Performance metrics dashboard
- Centralized documentation hub with links to governance and research collateral

## Deployment

The site automatically deploys to GitHub Pages via GitHub Actions when changes are pushed to the main branch.