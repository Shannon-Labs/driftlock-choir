#!/bin/bash

# DriftLock Website Deployment Script
# This script deploys the landing page to various hosting platforms

set -e

echo "🚀 DriftLock Website Deployment"
echo "================================"

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: index.html not found. Please run from the project root."
    exit 1
fi

# Create deployment directory
DEPLOY_DIR="deploy"
echo "📁 Creating deployment directory: $DEPLOY_DIR"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy website files
echo "📋 Copying website files..."
cp index.html $DEPLOY_DIR/
cp -r patent_fig*.png $DEPLOY_DIR/ 2>/dev/null || echo "⚠️  Warning: Patent figures not found"
cp patent/PATENT_SUPPORTING_DOCUMENTATION.html $DEPLOY_DIR/ 2>/dev/null || echo "⚠️  Warning: Supporting documentation not found"

# Create a simple README for deployment
cat > $DEPLOY_DIR/README.md << EOF
# DriftLock Landing Page

This directory contains the DriftLock landing page files for deployment.

## Files
- \`index.html\` - Main landing page
- \`patent_fig*.png\` - Technical diagrams
- \`PATENT_SUPPORTING_DOCUMENTATION.html\` - Interactive technical documentation

## Deployment Options

### GitHub Pages
1. Push this directory to a GitHub repository
2. Enable GitHub Pages in repository settings
3. Set source to main branch

### Netlify
1. Drag and drop this directory to netlify.com
2. Or connect your GitHub repository

### Vercel
1. Install Vercel CLI: \`npm i -g vercel\`
2. Run: \`vercel --prod\`

### Custom Domain
Update the links in index.html to point to your domain.
EOF

echo "✅ Deployment files created in: $DEPLOY_DIR"
echo ""
echo "🌐 Deployment Options:"
echo "1. GitHub Pages: Push $DEPLOY_DIR to a GitHub repo and enable Pages"
echo "2. Netlify: Drag $DEPLOY_DIR to netlify.com"
echo "3. Vercel: Run 'vercel --prod' from $DEPLOY_DIR"
echo "4. Custom hosting: Upload $DEPLOY_DIR contents to your web server"
echo ""
echo "📧 For custom domain setup, update links in index.html"
echo "🔗 Current GitHub repo link: https://github.com/yourusername/driftlock choir"
echo ""
echo "✨ Deployment package ready!"
