// Getting Started JavaScript for interactive tutorials

class GettingStartedGuide {
  constructor() {
    this.currentStep = 1;
    this.requirements = {
      python: false,
      pip: false,
      git: false
    };
    
    this.initializeEventListeners();
    this.initializeStepNavigation();
    this.initializeCodeExamples();
  }
  
  initializeEventListeners() {
    // Learning path buttons
    document.querySelectorAll('.path-button').forEach(button => {
      button.addEventListener('click', (e) => {
        const path = e.target.dataset.path;
        this.selectLearningPath(path);
      });
    });
    
    // Installation step navigation
    document.querySelectorAll('.step').forEach(step => {
      step.addEventListener('click', (e) => {
        const stepNumber = parseInt(e.currentTarget.dataset.step);
        this.goToStep(stepNumber);
      });
    });
    
    // Option tabs for dependency installation
    document.querySelectorAll('.option-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        const option = e.target.dataset.option;
        this.switchInstallOption(option);
      });
    });
  }
  
  initializeStepNavigation() {
    // Enable step 1 by default
    this.updateStepState(1, true);
  }
  
  initializeCodeExamples() {
    // Initialize syntax highlighting if Prism is available
    if (window.Prism) {
      Prism.highlightAll();
    }
  }
  
  selectLearningPath(path) {
    // Highlight selected path
    document.querySelectorAll('.path-card').forEach(card => {
      card.classList.remove('selected');
    });
    
    event.target.closest('.path-card').classList.add('selected');
    
    // Customize content based on path
    switch (path) {
      case 'researcher':
        this.customizeForResearcher();
        break;
      case 'developer':
        this.customizeForDeveloper();
        break;
      case 'explorer':
        this.customizeForExplorer();
        break;
    }
    
    // Smooth scroll to installation section
    document.querySelector('.installation-section').scrollIntoView({
      behavior: 'smooth'
    });
  }
  
  customizeForResearcher() {
    // Add researcher-specific content
    console.log('Customizing for researcher path');
  }
  
  customizeForDeveloper() {
    // Add developer-specific content
    console.log('Customizing for developer path');
  }
  
  customizeForExplorer() {
    // Add explorer-specific content
    console.log('Customizing for explorer path');
  }
  
  goToStep(stepNumber) {
    if (stepNumber <= this.currentStep || this.isStepUnlocked(stepNumber)) {
      this.showStep(stepNumber);
      this.currentStep = stepNumber;
    }
  }
  
  showStep(stepNumber) {
    // Update step buttons
    document.querySelectorAll('.step').forEach(step => {
      step.classList.remove('active');
    });
    document.querySelector(`[data-step="${stepNumber}"]`).classList.add('active');
    
    // Update step content
    document.querySelectorAll('.step-detail').forEach(detail => {
      detail.classList.remove('active');
    });
    document.getElementById(`step-${stepNumber}-detail`).classList.add('active');
  }
  
  nextStep(stepNumber) {
    if (stepNumber <= 4) {
      this.showStep(stepNumber);
      this.currentStep = stepNumber;
      this.updateStepState(stepNumber, true);
    }
  }
  
  updateStepState(stepNumber, enabled) {
    const step = document.querySelector(`[data-step="${stepNumber}"]`);
    if (enabled) {
      step.classList.add('enabled');
    } else {
      step.classList.remove('enabled');
    }
  }
  
  isStepUnlocked(stepNumber) {
    return stepNumber <= this.currentStep;
  }
  
  switchInstallOption(option) {
    // Update tab buttons
    document.querySelectorAll('.option-tab').forEach(tab => {
      tab.classList.remove('active');
    });
    document.querySelector(`[data-option="${option}"]`).classList.add('active');
    
    // Update option content
    document.querySelectorAll('.option-content').forEach(content => {
      content.classList.remove('active');
    });
    document.getElementById(`${option}-option`).classList.add('active');
  }
  
  // Simulated requirement checks
  checkRequirement(requirement) {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simulate success (in real implementation, this would check actual system)
        this.requirements[requirement] = true;
        resolve(true);
      }, 1000 + Math.random() * 1000);
    });
  }
  
  updateRequirementUI(requirement, status, success) {
    const icon = document.getElementById(`${requirement}-check`);
    const statusElement = document.getElementById(`${requirement}-status`);
    
    if (status === 'checking') {
      icon.textContent = '⏳';
      statusElement.textContent = 'Checking...';
    } else if (success) {
      icon.textContent = '✅';
      statusElement.textContent = 'Found';
      statusElement.style.color = '#10b981';
    } else {
      icon.textContent = '❌';
      statusElement.textContent = 'Not found';
      statusElement.style.color = '#ef4444';
    }
    
    this.checkAllRequirements();
  }
  
  checkAllRequirements() {
    const allMet = Object.values(this.requirements).every(req => req);
    const nextButton = document.getElementById('step-1-next');
    
    if (allMet && nextButton) {
      nextButton.disabled = false;
    }
  }
}

// Global functions for button callbacks
function checkPython() {
  guide.updateRequirementUI('python', 'checking');
  guide.checkRequirement('python').then(success => {
    guide.updateRequirementUI('python', 'complete', success);
  });
}

function checkPip() {
  guide.updateRequirementUI('pip', 'checking');
  guide.checkRequirement('pip').then(success => {
    guide.updateRequirementUI('pip', 'complete', success);
  });
}

function checkGit() {
  guide.updateRequirementUI('git', 'checking');
  guide.checkRequirement('git').then(success => {
    guide.updateRequirementUI('git', 'complete', success);
  });
}

function nextStep(stepNumber) {
  guide.nextStep(stepNumber);
}

function copyToClipboard(elementId) {
  const element = document.getElementById(elementId);
  const text = element.textContent;
  
  navigator.clipboard.writeText(text).then(() => {
    // Show feedback
    const button = event.target.closest('.copy-button');
    const originalIcon = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i>';
    button.style.color = '#10b981';
    
    setTimeout(() => {
      button.innerHTML = originalIcon;
      button.style.color = '';
    }, 2000);
  });
}

function simulateInstallation() {
  const progressBar = document.getElementById('install-progress');
  const statusElement = document.getElementById('install-status');
  
  const packages = [
    'numpy',
    'scipy',
    'matplotlib',
    'pytest',
    'pydantic',
    'typing-extensions'
  ];
  
  let progress = 0;
  let packageIndex = 0;
  
  const installInterval = setInterval(() => {
    if (packageIndex < packages.length) {
      statusElement.textContent = `Installing ${packages[packageIndex]}...`;
      progress += (100 / packages.length);
      progressBar.style.width = `${progress}%`;
      packageIndex++;
    } else {
      clearInterval(installInterval);
      statusElement.textContent = 'Installation complete!';
      statusElement.style.color = '#10b981';
      progressBar.style.background = '#10b981';
    }
  }, 800);
}

function runExperimentSimulation() {
  const outputElement = document.getElementById('experiment-output');
  const successElement = document.getElementById('success-message');
  
  outputElement.style.display = 'block';
  
  // Simulate experiment running
  const lines = outputElement.querySelectorAll('.terminal-line');
  
  lines.forEach((line, index) => {
    setTimeout(() => {
      line.style.opacity = '1';
      line.style.transform = 'translateX(0)';
      
      if (index === lines.length - 1) {
        setTimeout(() => {
          successElement.style.display = 'block';
          successElement.scrollIntoView({ behavior: 'smooth' });
        }, 1000);
      }
    }, index * 500);
  });
}

function runCodeExample(example) {
  const outputElement = document.getElementById(`${example}-output`);
  
  if (outputElement) {
    outputElement.style.display = 'block';
    
    // Simulate processing
    setTimeout(() => {
      outputElement.querySelector('.output-content').style.opacity = '1';
    }, 500);
  }
}

function modifyExample(example) {
  alert('Parameter modification interface would open here');
}

function visualizeConsensus() {
  const vizElement = document.getElementById('consensus-viz');
  const canvas = document.getElementById('consensus-canvas');
  
  if (vizElement && canvas) {
    vizElement.style.display = 'block';
    drawConsensusVisualization(canvas);
  }
}

function drawConsensusVisualization(canvas) {
  const ctx = canvas.getContext('2d');
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw convergence curves
  const iterations = 20;
  const node1Data = [];
  const node2Data = [];
  
  for (let i = 0; i < iterations; i++) {
    const progress = i / iterations;
    node1Data.push(100 + 50 * Math.exp(-i / 5) * Math.sin(i * 0.5));
    node2Data.push(150 - 50 * Math.exp(-i / 5) * Math.cos(i * 0.3));
  }
  
  // Draw grid
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  
  // Draw node 1 convergence
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.beginPath();
  node1Data.forEach((value, i) => {
    const x = (i / (iterations - 1)) * canvas.width;
    const y = canvas.height - ((value - 50) / 150) * canvas.height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Draw node 2 convergence
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.beginPath();
  node2Data.forEach((value, i) => {
    const x = (i / (iterations - 1)) * canvas.width;
    const y = canvas.height - ((value - 50) / 150) * canvas.height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Add labels
  ctx.fillStyle = '#374151';
  ctx.font = '12px Inter';
  ctx.fillText('Node 1', 10, 20);
  ctx.fillStyle = '#3b82f6';
  ctx.fillRect(60, 15, 10, 2);
  
  ctx.fillStyle = '#374151';
  ctx.fillText('Node 2', 10, 35);
  ctx.fillStyle = '#ef4444';
  ctx.fillRect(60, 30, 10, 2);
}

// Global instance
let guide;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  guide = new GettingStartedGuide();
  
  // Add CSS for terminal animation
  const style = document.createElement('style');
  style.textContent = `
    .terminal-line {
      opacity: 0;
      transform: translateX(-10px);
      transition: all 0.3s ease;
    }
    
    .terminal-line.success {
      color: #10b981;
    }
    
    .highlight {
      color: #f59e0b;
      font-weight: 700;
    }
    
    .path-card.selected {
      border: 2px solid #10b981;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    .step.enabled {
      cursor: pointer;
    }
    
    .step:not(.enabled) {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .progress-bar {
      width: 100%;
      height: 8px;
      background: #e5e7eb;
      border-radius: 4px;
      overflow: hidden;
      margin: 1rem 0;
    }
    
    .progress-fill {
      height: 100%;
      background: #3b82f6;
      width: 0%;
      transition: width 0.3s ease;
    }
  `;
  document.head.appendChild(style);
});