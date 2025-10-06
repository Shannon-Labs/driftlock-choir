// Interactive Demo JavaScript for Chronometric Interferometry Simulator

class ChronometricSimulator {
  constructor() {
    this.parameters = {
      frequencyOffset: 100, // Hz
      timeDelay: 100, // ps
      snr: 30, // dB
      duration: 10, // ms
      samplingRate: 20000 // Hz
    };
    
    this.isPlaying = false;
    this.animationFrame = null;
    this.currentTab = 'waveform';
    this.zoomLevel = 1;
    
    this.waveformData = null;
    this.spectrumData = null;
    this.phaseData = null;
    this.convergenceData = null;
    
    this.initializeControls();
    this.initializeTabs();
    this.initializeCharts();
    this.runInitialSimulation();
  }
  
  initializeControls() {
    // Create parameter sliders
    this.createSlider('frequency-offset-control', {
      label: 'Frequency Offset (Δf)',
      min: 10,
      max: 500,
      step: 10,
      value: this.parameters.frequencyOffset,
      unit: ' Hz',
      onChange: (value) => {
        this.parameters.frequencyOffset = value;
        this.updateLiveCalculations();
      }
    });
    
    this.createSlider('time-delay-control', {
      label: 'Time-of-Flight Delay (τ)',
      min: 10,
      max: 1000,
      step: 10,
      value: this.parameters.timeDelay,
      unit: ' ps',
      onChange: (value) => {
        this.parameters.timeDelay = value;
        this.updateLiveCalculations();
      }
    });
    
    this.createSlider('snr-control', {
      label: 'Signal-to-Noise Ratio',
      min: 10,
      max: 50,
      step: 1,
      value: this.parameters.snr,
      unit: ' dB',
      onChange: (value) => {
        this.parameters.snr = value;
        this.updateLiveCalculations();
      }
    });
    
    this.createSlider('duration-control', {
      label: 'Analysis Duration',
      min: 5,
      max: 100,
      step: 5,
      value: this.parameters.duration,
      unit: ' ms',
      onChange: (value) => {
        this.parameters.duration = value;
      }
    });
    
    this.createSlider('sampling-rate-control', {
      label: 'Sampling Rate',
      min: 1000,
      max: 50000,
      step: 1000,
      value: this.parameters.samplingRate,
      unit: ' Hz',
      onChange: (value) => {
        this.parameters.samplingRate = value;
      }
    });
    
    // Button event listeners
    document.getElementById('run-simulation').addEventListener('click', () => {
      this.runSimulation();
    });
    
    document.getElementById('reset-parameters').addEventListener('click', () => {
      this.resetParameters();
    });
    
    document.getElementById('export-data').addEventListener('click', () => {
      this.exportData();
    });
    
    document.getElementById('play-pause').addEventListener('click', () => {
      this.toggleAnimation();
    });
    
    document.getElementById('zoom-in').addEventListener('click', () => {
      this.zoomLevel *= 1.5;
      this.redrawCurrentChart();
    });
    
    document.getElementById('zoom-out').addEventListener('click', () => {
      this.zoomLevel /= 1.5;
      this.redrawCurrentChart();
    });
  }
  
  createSlider(containerId, config) {
    return window.DriftlockChoir.createParameterSlider(
      document.getElementById(containerId),
      config
    );
  }
  
  initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const tabName = e.target.dataset.tab;
        this.switchTab(tabName);
      });
    });
  }
  
  switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(btn => {
      btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update active tab content
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    this.currentTab = tabName;
    this.redrawCurrentChart();
  }
  
  initializeCharts() {
    // Initialize canvas contexts
    this.waveformCtx = document.getElementById('waveform-canvas').getContext('2d');
    this.spectrumCtx = document.getElementById('spectrum-canvas').getContext('2d');
    this.phaseCtx = document.getElementById('phase-canvas').getContext('2d');
    this.convergenceCtx = document.getElementById('convergence-canvas').getContext('2d');
    this.benchmarkCtx = document.getElementById('benchmark-canvas').getContext('2d');
    
    this.drawBenchmarkChart();
  }
  
  runInitialSimulation() {
    this.updateLiveCalculations();
    this.runSimulation();
  }
  
  runSimulation() {
    const loader = window.DriftlockChoir.createLoader(
      document.querySelector('.visualization-panel')
    );
    loader.show();
    
    // Simulate computation delay
    setTimeout(() => {
      this.generateSimulationData();
      this.updatePerformanceMetrics();
      this.redrawCurrentChart();
      loader.hide();
      loader.remove();
    }, 500);
  }
  
  generateSimulationData() {
    const { frequencyOffset, timeDelay, snr, duration, samplingRate } = this.parameters;
    
    // Generate time vector
    const numSamples = Math.floor(duration * samplingRate / 1000);
    const timeVector = Array.from({length: numSamples}, (_, i) => i / samplingRate);
    
    // Generate beat-note waveform
    const carrierFreq = 440; // Hz (for audio-like demonstration)
    const beatFreq = frequencyOffset;
    const noiseLevel = Math.pow(10, -snr / 20);
    
    this.waveformData = {
      time: timeVector,
      signal: timeVector.map(t => {
        const carrier = Math.sin(2 * Math.PI * carrierFreq * t);
        const beat = 0.5 * (1 + Math.sin(2 * Math.PI * beatFreq * t));
        const phase = 2 * Math.PI * timeDelay * 1e-12 * carrierFreq;
        const noise = (Math.random() - 0.5) * 2 * noiseLevel;
        return carrier * beat * Math.cos(phase) + noise;
      }),
      envelope: timeVector.map(t => 0.5 * (1 + Math.sin(2 * Math.PI * beatFreq * t)))
    };
    
    // Generate frequency spectrum
    this.generateSpectrumData();
    
    // Generate phase data
    this.generatePhaseData();
    
    // Generate convergence data
    this.generateConvergenceData();
    
    // Update information displays
    this.updateInfoDisplays();
  }
  
  generateSpectrumData() {
    const N = 1024;
    const fs = this.parameters.samplingRate;
    
    // Simple peak detection simulation
    const peaks = [
      { frequency: 440, magnitude: 0.8, label: 'Carrier' },
      { frequency: 440 + this.parameters.frequencyOffset, magnitude: 0.4, label: 'Beat+' },
      { frequency: Math.abs(440 - this.parameters.frequencyOffset), magnitude: 0.4, label: 'Beat-' },
      { frequency: this.parameters.frequencyOffset, magnitude: 0.6, label: 'Beat Frequency' }
    ];
    
    this.spectrumData = {
      frequencies: Array.from({length: N/2}, (_, i) => i * fs / N),
      magnitudes: Array.from({length: N/2}, (_, i) => {
        const freq = i * fs / N;
        let magnitude = 0.1 * Math.random(); // noise floor
        
        peaks.forEach(peak => {
          const distance = Math.abs(freq - peak.frequency);
          if (distance < 10) {
            magnitude += peak.magnitude * Math.exp(-distance * distance / 50);
          }
        });
        
        return magnitude;
      }),
      peaks: peaks.filter(p => p.frequency < fs/2)
    };
  }
  
  generatePhaseData() {
    const { timeDelay, frequencyOffset } = this.parameters;
    const timeVector = this.waveformData.time;
    
    this.phaseData = {
      time: timeVector,
      instantaneousPhase: timeVector.map(t => {
        const phase = 2 * Math.PI * frequencyOffset * t + 
                     2 * Math.PI * timeDelay * 1e-12 * frequencyOffset;
        return (phase % (2 * Math.PI)) - Math.PI; // wrap to [-π, π]
      }),
      phaseSlope: 2 * Math.PI * frequencyOffset,
      extractedTau: timeDelay + (Math.random() - 0.5) * 2, // add small error
      extractedDeltaF: frequencyOffset + (Math.random() - 0.5) * 0.1
    };
  }
  
  generateConvergenceData() {
    const iterations = 50;
    const targetTau = this.parameters.timeDelay;
    const targetDeltaF = this.parameters.frequencyOffset;
    
    this.convergenceData = {
      iterations: Array.from({length: iterations}, (_, i) => i + 1),
      tauEstimates: Array.from({length: iterations}, (_, i) => {
        const progress = (i + 1) / iterations;
        const error = (1 - progress) * 20 + Math.random() * 2;
        return targetTau + error * Math.exp(-i / 10);
      }),
      deltaFEstimates: Array.from({length: iterations}, (_, i) => {
        const progress = (i + 1) / iterations;
        const error = (1 - progress) * 5 + Math.random() * 0.5;
        return targetDeltaF + error * Math.exp(-i / 8);
      }),
      rmseHistory: Array.from({length: iterations}, (_, i) => {
        const baseRMSE = this.calculatePrecision();
        return baseRMSE * Math.exp(-i / 15) + Math.random() * 0.5;
      })
    };
  }
  
  updateInfoDisplays() {
    // Waveform info
    document.getElementById('beat-frequency').textContent = 
      this.parameters.frequencyOffset.toFixed(1);
    document.getElementById('amplitude-mod').textContent = 
      (50 + Math.random() * 10).toFixed(1);
    document.getElementById('phase-coherence').textContent = 
      (0.85 + Math.random() * 0.1).toFixed(3);
    
    // Peak list
    const peakList = document.getElementById('peak-list');
    peakList.innerHTML = this.spectrumData.peaks.map(peak => 
      `<div class="peak-item">
        <strong>${peak.label}:</strong> ${peak.frequency.toFixed(1)} Hz 
        (${(20 * Math.log10(peak.magnitude)).toFixed(1)} dB)
      </div>`
    ).join('');
    
    // Phase metrics
    document.getElementById('phase-slope').textContent = 
      this.phaseData.phaseSlope.toFixed(2);
    document.getElementById('phase-variance').textContent = 
      (0.1 + Math.random() * 0.05).toFixed(4);
    
    // Convergence metrics
    const convergenceTime = Math.max(10, 100 - this.parameters.snr * 2);
    document.getElementById('convergence-time').textContent = 
      convergenceTime.toFixed(0);
    document.getElementById('final-rmse').textContent = 
      this.calculatePrecision().toFixed(1);
  }
  
  calculatePrecision() {
    // Simplified precision calculation based on SNR and frequency offset
    const { snr, frequencyOffset } = this.parameters;
    const baselinePrecision = 100; // ps
    const snrFactor = Math.pow(10, snr / 20);
    const freqFactor = frequencyOffset / 100;
    
    return Math.max(0.5, baselinePrecision / (snrFactor * freqFactor));
  }
  
  updatePerformanceMetrics() {
    const timingPrecision = this.calculatePrecision();
    const frequencyAccuracy = Math.max(0.1, 10 / Math.pow(10, this.parameters.snr / 20));
    
    document.getElementById('timing-precision').textContent = timingPrecision.toFixed(1);
    document.getElementById('frequency-accuracy').textContent = frequencyAccuracy.toFixed(1);
    
    // Update demo table values
    document.getElementById('demo-timing').textContent = `${timingPrecision.toFixed(1)} ps`;
    document.getElementById('demo-frequency').textContent = `${frequencyAccuracy.toFixed(1)} ppb`;
  }
  
  updateLiveCalculations() {
    const { frequencyOffset, timeDelay, snr } = this.parameters;
    
    // Calculate live values
    const deltaPhi = 2 * Math.PI * timeDelay * 1e-12 * frequencyOffset;
    const tau = (deltaPhi / (2 * Math.PI * frequencyOffset)) * 1e12; // convert to ps
    const sigmaPhi = 0.1; // assumed phase noise
    const snrLinear = Math.pow(10, snr / 20);
    const sigmaTau = (sigmaPhi / (2 * Math.PI * frequencyOffset)) / Math.sqrt(snrLinear) * 1e12;
    
    // Update displays
    document.getElementById('live-delta-phi').textContent = deltaPhi.toFixed(3);
    document.getElementById('live-delta-f').textContent = frequencyOffset.toFixed(1);
    document.getElementById('live-tau').textContent = tau.toFixed(1);
    document.getElementById('live-f2').textContent = (2.4000000 + frequencyOffset / 1e9).toFixed(7);
    document.getElementById('live-fbeat').textContent = frequencyOffset.toFixed(1);
    document.getElementById('live-sigma-phi').textContent = sigmaPhi.toFixed(3);
    document.getElementById('live-snr').textContent = snr.toFixed(1);
    document.getElementById('live-sigma-tau').textContent = sigmaTau.toFixed(1);
  }
  
  redrawCurrentChart() {
    switch (this.currentTab) {
      case 'waveform':
        this.drawWaveform();
        break;
      case 'spectrum':
        this.drawSpectrum();
        break;
      case 'phase':
        this.drawPhase();
        break;
      case 'convergence':
        this.drawConvergence();
        break;
    }
  }
  
  drawWaveform() {
    if (!this.waveformData) return;
    
    const canvas = document.getElementById('waveform-canvas');
    const ctx = this.waveformCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Setup
    const margin = 60;
    const plotWidth = canvas.width - 2 * margin;
    const plotHeight = canvas.height - 2 * margin;
    
    // Draw grid
    this.drawGrid(ctx, margin, plotWidth, plotHeight);
    
    // Draw signal
    const timeData = this.waveformData.time;
    const signalData = this.waveformData.signal;
    const envelopeData = this.waveformData.envelope;
    
    const maxTime = Math.max(...timeData) / this.zoomLevel;
    const minSignal = -1.2;
    const maxSignal = 1.2;
    
    // Draw envelope
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    timeData.forEach((t, i) => {
      if (t > maxTime) return;
      
      const x = margin + (t / maxTime) * plotWidth;
      const y = margin + plotHeight - ((envelopeData[i] - minSignal) / (maxSignal - minSignal)) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Draw negative envelope
    ctx.beginPath();
    timeData.forEach((t, i) => {
      if (t > maxTime) return;
      
      const x = margin + (t / maxTime) * plotWidth;
      const y = margin + plotHeight - ((-envelopeData[i] - minSignal) / (maxSignal - minSignal)) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Draw main signal
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    timeData.forEach((t, i) => {
      if (t > maxTime) return;
      
      const x = margin + (t / maxTime) * plotWidth;
      const y = margin + plotHeight - ((signalData[i] - minSignal) / (maxSignal - minSignal)) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Draw axes labels
    this.drawAxesLabels(ctx, canvas.width, canvas.height, 'Time (ms)', 'Amplitude');
  }
  
  drawSpectrum() {
    if (!this.spectrumData) return;
    
    const canvas = document.getElementById('spectrum-canvas');
    const ctx = this.spectrumCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const margin = 60;
    const plotWidth = canvas.width - 2 * margin;
    const plotHeight = canvas.height - 2 * margin;
    
    this.drawGrid(ctx, margin, plotWidth, plotHeight);
    
    // Draw spectrum
    const frequencies = this.spectrumData.frequencies;
    const magnitudes = this.spectrumData.magnitudes;
    const maxFreq = Math.min(1000, Math.max(...frequencies)) / this.zoomLevel;
    const maxMag = Math.max(...magnitudes);
    
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    frequencies.forEach((freq, i) => {
      if (freq > maxFreq) return;
      
      const x = margin + (freq / maxFreq) * plotWidth;
      const y = margin + plotHeight - (magnitudes[i] / maxMag) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Highlight peaks
    this.spectrumData.peaks.forEach(peak => {
      if (peak.frequency > maxFreq) return;
      
      const x = margin + (peak.frequency / maxFreq) * plotWidth;
      const y = margin + plotHeight - (peak.magnitude / maxMag) * plotHeight;
      
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      
      // Label
      ctx.fillStyle = '#374151';
      ctx.font = '12px Inter';
      ctx.fillText(peak.label, x + 5, y - 5);
    });
    
    this.drawAxesLabels(ctx, canvas.width, canvas.height, 'Frequency (Hz)', 'Magnitude');
  }
  
  drawPhase() {
    if (!this.phaseData) return;
    
    const canvas = document.getElementById('phase-canvas');
    const ctx = this.phaseCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const margin = 60;
    const plotWidth = canvas.width - 2 * margin;
    const plotHeight = canvas.height - 2 * margin;
    
    this.drawGrid(ctx, margin, plotWidth, plotHeight);
    
    // Draw phase evolution
    const timeData = this.phaseData.time;
    const phaseData = this.phaseData.instantaneousPhase;
    const maxTime = Math.max(...timeData) / this.zoomLevel;
    
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    timeData.forEach((t, i) => {
      if (t > maxTime) return;
      
      const x = margin + (t / maxTime) * plotWidth;
      const y = margin + plotHeight - ((phaseData[i] + Math.PI) / (2 * Math.PI)) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    this.drawAxesLabels(ctx, canvas.width, canvas.height, 'Time (ms)', 'Phase (rad)');
  }
  
  drawConvergence() {
    if (!this.convergenceData) return;
    
    const canvas = document.getElementById('convergence-canvas');
    const ctx = this.convergenceCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const margin = 60;
    const plotWidth = canvas.width - 2 * margin;
    const plotHeight = canvas.height - 2 * margin;
    
    this.drawGrid(ctx, margin, plotWidth, plotHeight);
    
    // Draw RMSE convergence
    const iterations = this.convergenceData.iterations;
    const rmseHistory = this.convergenceData.rmseHistory;
    const maxIter = Math.max(...iterations);
    const maxRMSE = Math.max(...rmseHistory);
    
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    iterations.forEach((iter, i) => {
      const x = margin + (iter / maxIter) * plotWidth;
      const y = margin + plotHeight - (rmseHistory[i] / maxRMSE) * plotHeight;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Draw target line
    const targetY = margin + plotHeight - (2.1 / maxRMSE) * plotHeight;
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin, targetY);
    ctx.lineTo(margin + plotWidth, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Label target
    ctx.fillStyle = '#10b981';
    ctx.font = '12px Inter';
    ctx.fillText('Target: 2.1 ps', margin + 10, targetY - 5);
    
    this.drawAxesLabels(ctx, canvas.width, canvas.height, 'Iterations', 'RMSE (ps)');
  }
  
  drawBenchmarkChart() {
    const canvas = document.getElementById('benchmark-canvas');
    const ctx = this.benchmarkCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw benchmark comparison bars
    const technologies = ['Driftlock', 'NTP', 'PTP', 'GPS'];
    const precisions = [2.1, 100000000, 1000000, 100000]; // ps
    const colors = ['#10b981', '#6b7280', '#9ca3af', '#d1d5db'];
    
    const margin = 60;
    const plotWidth = canvas.width - 2 * margin;
    const plotHeight = canvas.height - 2 * margin;
    const barWidth = plotWidth / technologies.length * 0.8;
    const barSpacing = plotWidth / technologies.length;
    
    const maxPrecision = Math.max(...precisions);
    
    technologies.forEach((tech, i) => {
      const x = margin + i * barSpacing + (barSpacing - barWidth) / 2;
      const height = (precisions[i] / maxPrecision) * plotHeight;
      const y = margin + plotHeight - height;
      
      ctx.fillStyle = colors[i];
      ctx.fillRect(x, y, barWidth, height);
      
      // Label
      ctx.fillStyle = '#374151';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(tech, x + barWidth / 2, margin + plotHeight + 20);
      
      // Value
      const value = precisions[i] < 1000 ? `${precisions[i]} ps` : 
                   precisions[i] < 1000000 ? `${(precisions[i]/1000).toFixed(0)} ns` :
                   precisions[i] < 1000000000 ? `${(precisions[i]/1000000).toFixed(0)} µs` :
                   `${(precisions[i]/1000000000).toFixed(0)} ms`;
      ctx.fillText(value, x + barWidth / 2, y - 5);
    });
    
    ctx.textAlign = 'left';
    
    // Y-axis label
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#374151';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Timing Precision (log scale)', 0, 0);
    ctx.restore();
  }
  
  drawGrid(ctx, margin, plotWidth, plotHeight) {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = margin + (i / 10) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, margin + plotHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = margin + (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(margin + plotWidth, y);
      ctx.stroke();
    }
  }
  
  drawAxesLabels(ctx, canvasWidth, canvasHeight, xLabel, yLabel) {
    ctx.fillStyle = '#374151';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    
    // X-axis label
    ctx.fillText(xLabel, canvasWidth / 2, canvasHeight - 20);
    
    // Y-axis label
    ctx.save();
    ctx.translate(20, canvasHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    ctx.textAlign = 'left';
  }
  
  toggleAnimation() {
    const button = document.getElementById('play-pause');
    const icon = button.querySelector('i');
    
    if (this.isPlaying) {
      this.isPlaying = false;
      icon.className = 'fas fa-play';
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
      }
    } else {
      this.isPlaying = true;
      icon.className = 'fas fa-pause';
      this.animate();
    }
  }
  
  animate() {
    if (!this.isPlaying) return;
    
    // Simple animation: shift waveform data
    if (this.waveformData && this.currentTab === 'waveform') {
      const shiftAmount = 0.001; // 1ms
      this.waveformData.time = this.waveformData.time.map(t => t + shiftAmount);
      this.drawWaveform();
    }
    
    this.animationFrame = requestAnimationFrame(() => this.animate());
  }
  
  resetParameters() {
    this.parameters = {
      frequencyOffset: 100,
      timeDelay: 100,
      snr: 30,
      duration: 10,
      samplingRate: 20000
    };
    
    // Reset all sliders
    document.querySelectorAll('.parameter-slider').forEach(slider => {
      const paramName = slider.closest('[id]').id.replace('-control', '').replace('-', '');
      const camelCaseName = paramName.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
      if (this.parameters[camelCaseName] !== undefined) {
        slider.value = this.parameters[camelCaseName];
        slider.dispatchEvent(new Event('input'));
      }
    });
    
    this.runSimulation();
  }
  
  exportData() {
    const data = {
      parameters: this.parameters,
      waveformData: this.waveformData,
      spectrumData: this.spectrumData,
      phaseData: this.phaseData,
      convergenceData: this.convergenceData,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chronometric_simulation_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

// Initialize the simulator when the page loads
document.addEventListener('DOMContentLoaded', function() {
  new ChronometricSimulator();
});