# 🚀 Driftlock Website Deployment Guide

This guide provides step-by-step instructions for deploying the Driftlock landing page to various hosting platforms.

## 📦 Pre-Deployment Setup

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Add professional landing page and documentation"
git push origin main
```

### 2. Run Deployment Script
```bash
# Generate deployment package
./deploy.sh
```

This creates a `deploy/` directory with all necessary files.

## 🌐 Deployment Options

### Option 1: GitHub Pages (Recommended for Open Source)

#### Setup
1. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / `docs` folder

2. **Deploy**:
   ```bash
   # Copy deploy files to docs folder
   mkdir -p docs
   cp -r deploy/* docs/
   git add docs/
   git commit -m "Deploy website to GitHub Pages"
   git push origin main
   ```

3. **Access**: `https://yourusername.github.io/driftlock choir`

#### Custom Domain (Optional)
1. Add `CNAME` file to docs folder:
   ```bash
   echo "driftlock choir.dev" > docs/CNAME
   ```
2. Configure DNS records with your domain provider

### Option 2: Netlify (Easiest)

#### Quick Deploy
1. Go to [netlify.com](https://netlify.com)
2. Drag and drop the `deploy/` folder to the deploy area
3. Your site will be live in seconds!

#### Continuous Deployment
1. Connect your GitHub repository to Netlify
2. Set build command: `./deploy.sh`
3. Set publish directory: `deploy`
4. Netlify will auto-deploy on every push

### Option 3: Vercel (Fast & Global)

#### CLI Deploy
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd deploy
vercel --prod
```

#### GitHub Integration
1. Connect repository at [vercel.com](https://vercel.com)
2. Set build command: `./deploy.sh`
3. Set output directory: `deploy`
4. Auto-deploys on push

### Option 4: Custom Hosting

#### Requirements
- Static file hosting (HTML, CSS, JS, images)
- HTTPS support recommended
- CDN for global performance

#### Upload Process
1. Upload contents of `deploy/` folder to your web server
2. Ensure `index.html` is in the root directory
3. Configure server to serve `index.html` for all routes (SPA routing)

## 🔧 Customization

### Update Repository Links
Edit `index.html` and replace:
- `https://github.com/yourusername/driftlock choir` with your actual repo URL
- Email addresses with your actual contact information

### Add Analytics
Add Google Analytics or other tracking:
```html
<!-- Add to <head> section -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Custom Domain Setup
1. **DNS Configuration**:
   - A record: `@` → hosting provider IP
   - CNAME: `www` → your domain

2. **SSL Certificate**:
   - Most hosting providers offer free SSL
   - Let's Encrypt for custom servers

## 📊 Performance Optimization

### Image Optimization
```bash
# Optimize PNG files (if imagemagick is installed)
convert patent/figures/patent_fig1_architecture.png -quality 85 -strip patent/figures/patent_fig1_architecture_opt.png
```

### Compression
Enable gzip compression on your web server:
```nginx
# Nginx example
gzip on;
gzip_types text/html text/css application/javascript image/png;
```

### CDN Setup
- Cloudflare (free tier available)
- AWS CloudFront
- Google Cloud CDN

## 🔍 SEO Optimization

### Meta Tags
The landing page includes:
- Open Graph tags for social sharing
- Twitter Card meta tags
- Proper title and description
- Canonical URL (update for custom domain)

### Sitemap
Create `sitemap.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://yourdomain.com/</loc>
    <lastmod>2025-01-20</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
```

## 📱 Mobile Optimization

The landing page is fully responsive and includes:
- Mobile-first design
- Touch-friendly navigation
- Optimized images for mobile
- Fast loading times

## 🔒 Security Considerations

### HTTPS
- Always use HTTPS in production
- Redirect HTTP to HTTPS
- Use HSTS headers

### Content Security Policy
Add to your server configuration:
```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src fonts.gstatic.com; script-src 'self' 'unsafe-inline'; img-src 'self' data:;">
```

## 📈 Monitoring & Analytics

### Performance Monitoring
- Google PageSpeed Insights
- GTmetrix
- WebPageTest

### Analytics Setup
- Google Analytics 4
- Hotjar for user behavior
- Search Console for SEO tracking

## 🚨 Troubleshooting

### Common Issues

#### Images Not Loading
- Check file paths are correct
- Ensure images are in the deploy directory
- Verify file permissions

#### Styling Issues
- Check browser console for errors
- Verify CSS is loading correctly
- Test in different browsers

#### Mobile Display Problems
- Test responsive design
- Check viewport meta tag
- Verify touch interactions

## 📞 Support

For deployment issues:
- **Technical Support**: dev@driftlock choir.dev
- **Hosting Questions**: hello@driftlock choir.dev
- **Custom Domain Setup**: support@driftlock choir.dev

## ✅ Deployment Checklist

- [ ] Repository is public and properly organized
- [ ] All files are committed and pushed
- [ ] Deployment script runs successfully
- [ ] Website loads correctly on chosen platform
- [ ] All links work properly
- [ ] Images display correctly
- [ ] Mobile responsiveness verified
- [ ] Analytics/tracking configured (optional)
- [ ] Custom domain configured (optional)
- [ ] SSL certificate active (if custom domain)

## 🎉 Success!

Your Driftlock landing page is now live and ready to impress investors and potential partners!

**Next Steps:**
1. Share the URL with your network
2. Submit to relevant directories
3. Set up monitoring and analytics
4. Plan content updates and improvements
