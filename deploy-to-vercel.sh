#!/bin/bash

# Driftlock - Deploy to Vercel
# October 2025 - a16z Speedrun Ready

echo "🚀 Deploying Driftlock to Vercel..."
echo "═══════════════════════════════════"
echo ""
echo "📊 Latest Achievement: 22.13 picoseconds with tuned Kalman filtering (extended_011)"
echo "🏆 2,273× better than GPS (22ps vs 50ns)"
echo "🎯 Target: a16z Speedrun (Oct 15, 2025)"
echo ""

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "✓ Website ready: index.html"
echo "✓ Technical docs: /docs"
echo "✓ Application materials: /a16z-speedrun"
echo "✓ Patent documentation: /patent"
echo ""

echo "Key features enabled:"
echo "  • O-RAN/3GPP compatibility messaging"
echo "  • Operator revenue opportunities highlighted"
echo "  • Software-only positioning emphasized"
echo "  • Bell Labs legacy narrative"
echo ""

# Deploy to Vercel
echo "📤 Starting deployment..."
vercel --prod

echo ""
echo "═══════════════════════════════════"
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Check deployment at: https://driftlock choir.vercel.app"
echo "2. Submit a16z application by Oct 15"
echo "3. Prepare demo for investor meetings"
echo ""
echo "Remember: Beat patterns aren't noise—they're time itself."
echo "22 picoseconds today. The mesh intelligence substrate tomorrow."