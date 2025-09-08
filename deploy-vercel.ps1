# Vercel Deployment Script for Virtual Notepad

Write-Host "🚀 Deploying Virtual Notepad to Vercel..." -ForegroundColor Green

# Check if we're in the right directory
if (!(Test-Path "web/index.html")) {
    Write-Host "❌ Error: web/index.html not found! Make sure you're in the project root." -ForegroundColor Red
    exit 1
}

# Check if Vercel CLI is installed
try {
    $vercelVersion = vercel --version
    Write-Host "✅ Vercel CLI found: $vercelVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Vercel CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g vercel
}

# Check if model conversion was done
if (Test-Path "models/gesture_model.h5") {
    Write-Host "📊 Model found. Converting to TensorFlow.js..." -ForegroundColor Blue
    try {
        & "d:/GitHub/Virtual-Notepad/.venv/Scripts/python.exe" convert_tfjs.py
    } catch {
        Write-Host "⚠️ Model conversion failed, deploying with demo mode" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ No trained model found, deploying with demo mode" -ForegroundColor Yellow
}

Write-Host "🌐 Deploying to Vercel..." -ForegroundColor Blue
vercel --prod

Write-Host "🎉 Deployment completed!" -ForegroundColor Green
Write-Host "📱 Your app should be available at the Vercel URL" -ForegroundColor Cyan
Write-Host "🔗 Visit https://vercel.com/dashboard to see your deployment" -ForegroundColor Cyan
