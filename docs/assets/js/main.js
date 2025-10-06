// Main JavaScript for Driftlock Choir GitHub Pages

document.addEventListener('DOMContentLoaded', function() {
  initializeAnimations();
  initializeInteractiveElements();
  initializeAudioControls();
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// Initialize animations
function initializeAnimations() {
  // Intersection Observer for scroll-triggered animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
        
        // Special handling for metrics animation
        if (entry.target.id === 'performance-metrics') {
          animateMetrics();
        }
      }
    });
  }, observerOptions);
  
  // Observe sections for animation
  document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
  });
}

// Animate performance metrics
function animateMetrics() {
  const metrics = document.querySelectorAll('.metric-value');
  
  metrics.forEach(metric => {
    const target = parseFloat(metric.dataset.target);
    const duration = 2000; // 2 seconds
    const steps = 60;
    const increment = target / steps;
    let current = 0;
    let step = 0;
    
    const timer = setInterval(() => {
      current += increment;
      step++;
      
      if (step >= steps) {
        current = target;
        clearInterval(timer);
      }
      
      // Format the number appropriately
      if (target >= 100) {
        metric.textContent = Math.round(current);
      } else {
        metric.textContent = current.toFixed(1);
      }
    }, duration / steps);
  });
}

// Initialize interactive elements
function initializeInteractiveElements() {
  // Simple beat-note visualization for preview canvas
  const canvas = document.getElementById('preview-canvas');
  if (canvas) {
    initBeatNotePreview(canvas);
  }
  
  // Navbar scroll effect
  window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
  
  // Add hover effects for metric cards
  const metricCards = document.querySelectorAll('.metric-card');
  metricCards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-8px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0) scale(1)';
    });
  });
}

// Beat-note visualization preview
function initBeatNotePreview(canvas) {
  const ctx = canvas.getContext('2d');
  let frame = 0;
  let isAnimating = true;
  
  function drawBeatNote() {
    if (!isAnimating) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x += 30) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
    }
    for (let y = 0; y < canvas.height; y += 30) {
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
    }
    ctx.stroke();
    
    // Draw beat pattern
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const centerY = canvas.height / 2;
    const amplitude = 30;
    
    for (let x = 0; x < canvas.width; x++) {
      const t = (x + frame * 1.5) * 0.02;
      
      // Carrier wave (high frequency)
      const carrier = Math.sin(t * 15);
      
      // Beat envelope (low frequency modulation)
      const beat = Math.sin(t * 1.5) * 0.5 + 0.5;
      
      // Combined signal
      const y = centerY + carrier * beat * amplitude;
      
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Draw beat envelope
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    for (let x = 0; x < canvas.width; x++) {
      const t = (x + frame * 1.5) * 0.02;
      const beat = Math.sin(t * 1.5) * 0.5 + 0.5;
      const y = centerY + beat * amplitude;
      
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Draw negative envelope
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
      const t = (x + frame * 1.5) * 0.02;
      const beat = Math.sin(t * 1.5) * 0.5 + 0.5;
      const y = centerY - beat * amplitude;
      
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    frame++;
    requestAnimationFrame(drawBeatNote);
  }
  
  // Start animation when canvas comes into view
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      isAnimating = entry.isIntersecting;
      if (isAnimating) {
        drawBeatNote();
      }
    });
  });
  
  observer.observe(canvas);
}

// Audio controls initialization
function initializeAudioControls() {
  const audioElements = document.querySelectorAll('audio');
  
  audioElements.forEach(audio => {
    // Add loading state
    audio.addEventListener('loadstart', function() {
      this.parentElement.classList.add('loading');
    });
    
    audio.addEventListener('canplay', function() {
      this.parentElement.classList.remove('loading');
    });
    
    // Add play/pause visual feedback
    audio.addEventListener('play', function() {
      this.parentElement.classList.add('playing');
    });
    
    audio.addEventListener('pause', function() {
      this.parentElement.classList.remove('playing');
    });
    
    audio.addEventListener('ended', function() {
      this.parentElement.classList.remove('playing');
    });
  });
}

// Utility functions for other pages
window.DriftlockChoir = {
  // Create parameter slider with real-time feedback
  createParameterSlider: function(container, config) {
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = config.min;
    slider.max = config.max;
    slider.step = config.step || 0.1;
    slider.value = config.value || config.min;
    slider.className = 'parameter-slider';
    
    const label = document.createElement('label');
    label.textContent = config.label;
    label.className = 'parameter-label';
    
    const valueDisplay = document.createElement('span');
    valueDisplay.textContent = slider.value + (config.unit || '');
    valueDisplay.className = 'parameter-value';
    
    slider.addEventListener('input', function() {
      valueDisplay.textContent = this.value + (config.unit || '');
      if (config.onChange) {
        config.onChange(parseFloat(this.value));
      }
    });
    
    const sliderContainer = document.createElement('div');
    sliderContainer.className = 'parameter-control';
    sliderContainer.appendChild(label);
    sliderContainer.appendChild(slider);
    sliderContainer.appendChild(valueDisplay);
    
    container.appendChild(sliderContainer);
    
    return {
      element: slider,
      getValue: () => parseFloat(slider.value),
      setValue: (val) => {
        slider.value = val;
        valueDisplay.textContent = val + (config.unit || '');
      }
    };
  },
  
  // Format numbers for display
  formatNumber: function(num, precision = 2) {
    if (num >= 1e9) {
      return (num / 1e9).toFixed(precision) + 'G';
    } else if (num >= 1e6) {
      return (num / 1e6).toFixed(precision) + 'M';
    } else if (num >= 1e3) {
      return (num / 1e3).toFixed(precision) + 'k';
    } else if (num < 0.001) {
      return num.toExponential(precision);
    } else {
      return num.toFixed(precision);
    }
  },
  
  // Create loading spinner
  createLoader: function(container) {
    const loader = document.createElement('div');
    loader.className = 'loader';
    loader.innerHTML = '<div class="spinner"></div><span>Computing...</span>';
    container.appendChild(loader);
    return {
      show: () => loader.style.display = 'flex',
      hide: () => loader.style.display = 'none',
      remove: () => loader.remove()
    };
  }
};

// CSS for loading states and animations
const style = document.createElement('style');
style.textContent = `
  .navbar.scrolled {
    background-color: rgba(30, 58, 138, 0.95) !important;
    backdrop-filter: blur(10px);
  }
  
  .animate-in {
    animation: slideInUp 0.8s ease-out;
  }
  
  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .audio-quick-play.loading::after {
    content: "Loading...";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(255, 255, 255, 0.9);
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }
  
  .audio-quick-play.playing {
    background: #f0f9ff;
    border-color: #3b82f6;
  }
  
  .parameter-control {
    margin: 1rem 0;
    padding: 1rem;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }
  
  .parameter-label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #374151;
  }
  
  .parameter-slider {
    width: 100%;
    margin: 0.5rem 0;
  }
  
  .parameter-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: #3b82f6;
    margin-left: 0.5rem;
  }
  
  .loader {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    flex-direction: column;
    gap: 1rem;
  }
  
  .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #e2e8f0;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

document.head.appendChild(style);