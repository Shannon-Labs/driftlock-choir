// Audio Laboratory JavaScript for interactive audio demonstrations

class AudioLaboratory {
  constructor() {
    this.audioContext = null;
    this.currentVisualization = {};
    this.synthesizer = null;
    
    this.initializeAudioContext();
    this.initializeVisualizations();
    this.initializeSynthesizer();
  }
  
  async initializeAudioContext() {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (error) {
      console.warn('Web Audio API not supported');
    }
  }
  
  initializeVisualizations() {
    // Initialize visualizations for each demo
    for (let i = 1; i <= 3; i++) {
      this.initializeDemoVisualization(i);
    }
  }
  
  initializeDemoVisualization(demoNumber) {
    const canvas = document.getElementById(`viz-canvas-${demoNumber}`);
    const audio = document.getElementById(`audio-${demoNumber}`);
    
    if (!canvas || !audio) return;
    
    const ctx = canvas.getContext('2d');
    
    // Create analyzer for real-time visualization
    if (this.audioContext) {
      const source = this.audioContext.createMediaElementSource(audio);
      const analyzer = this.audioContext.createAnalyser();
      
      analyzer.fftSize = 1024;
      source.connect(analyzer);
      analyzer.connect(this.audioContext.destination);
      
      this.currentVisualization[demoNumber] = {
        canvas,
        ctx,
        analyzer,
        isPlaying: false,
        animationFrame: null
      };
      
      // Audio event listeners
      audio.addEventListener('play', () => {
        this.startVisualization(demoNumber);
      });
      
      audio.addEventListener('pause', () => {
        this.stopVisualization(demoNumber);
      });
      
      audio.addEventListener('ended', () => {
        this.stopVisualization(demoNumber);
      });
    }
    
    // Draw initial waveform
    this.drawStaticWaveform(demoNumber);
  }
  
  drawStaticWaveform(demoNumber) {
    const viz = this.currentVisualization[demoNumber];
    if (!viz) return;
    
    const { canvas, ctx } = viz;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background grid
    this.drawGrid(ctx, canvas.width, canvas.height);
    
    // Generate sample waveform based on demo type
    const sampleData = this.generateSampleWaveform(demoNumber);
    this.drawWaveform(ctx, canvas.width, canvas.height, sampleData);
  }
  
  generateSampleWaveform(demoNumber) {
    const length = 400;
    const data = new Array(length);
    
    for (let i = 0; i < length; i++) {
      const t = i / length * 4 * Math.PI; // 2 cycles
      
      switch (demoNumber) {
        case 1: // Beat-note formation
          const carrier = Math.sin(t * 10);
          const beat = 0.5 * (1 + Math.sin(t * 2));
          data[i] = carrier * beat;
          break;
          
        case 2: // Chronomagnetic pulses
          const pulse1 = Math.sin(t * 3) * (Math.sin(t * 0.5) > 0 ? 1 : 0);
          const pulse2 = Math.sin(t * 6) * (Math.sin(t * 0.3) > 0 ? 0.5 : 0);
          data[i] = pulse1 + pulse2;
          break;
          
        case 3: // τ/Δf modulation
          const baseCarrier = Math.sin(t * 8);
          const tauMod = 0.3 * Math.sin(t * 0.5);
          const deltaFMod = 0.2 * Math.sin(t * 0.8);
          data[i] = baseCarrier * (1 + tauMod + deltaFMod);
          break;
          
        default:
          data[i] = Math.sin(t);
      }
    }
    
    return data;
  }
  
  startVisualization(demoNumber) {
    const viz = this.currentVisualization[demoNumber];
    if (!viz || viz.isPlaying) return;
    
    viz.isPlaying = true;
    this.visualizeAudio(demoNumber);
  }
  
  stopVisualization(demoNumber) {
    const viz = this.currentVisualization[demoNumber];
    if (!viz) return;
    
    viz.isPlaying = false;
    if (viz.animationFrame) {
      cancelAnimationFrame(viz.animationFrame);
    }
    
    // Return to static waveform
    this.drawStaticWaveform(demoNumber);
  }
  
  visualizeAudio(demoNumber) {
    const viz = this.currentVisualization[demoNumber];
    if (!viz || !viz.isPlaying) return;
    
    const { canvas, ctx, analyzer } = viz;
    
    // Get frequency data
    const bufferLength = analyzer.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyzer.getByteFrequencyData(dataArray);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.drawGrid(ctx, canvas.width, canvas.height);
    
    // Draw frequency spectrum
    this.drawSpectrum(ctx, canvas.width, canvas.height, dataArray);
    
    viz.animationFrame = requestAnimationFrame(() => {
      this.visualizeAudio(demoNumber);
    });
  }
  
  drawGrid(ctx, width, height) {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Horizontal lines
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Vertical lines
    for (let i = 0; i <= 8; i++) {
      const x = (i / 8) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }
  
  drawWaveform(ctx, width, height, data) {
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const centerY = height / 2;
    const amplitude = height * 0.4;
    
    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * width;
      const y = centerY + data[i] * amplitude;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
  }
  
  drawSpectrum(ctx, width, height, data) {
    const barWidth = width / data.length * 2;
    
    ctx.fillStyle = '#8b5cf6';
    
    for (let i = 0; i < data.length / 2; i++) {
      const barHeight = (data[i] / 255) * height;
      const x = i * barWidth;
      const y = height - barHeight;
      
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    }
  }
  
  initializeSynthesizer() {
    this.synthesizer = new AudioSynthesizer();
    
    // Control event listeners
    const carrierFreq = document.getElementById('carrier-freq');
    const beatFreq = document.getElementById('beat-freq');
    const generateBtn = document.getElementById('generate-audio');
    
    if (carrierFreq) {
      carrierFreq.addEventListener('input', (e) => {
        document.getElementById('carrier-freq-value').textContent = e.target.value + ' Hz';
        this.updateSynthesisVisualization();
      });
    }
    
    if (beatFreq) {
      beatFreq.addEventListener('input', (e) => {
        document.getElementById('beat-freq-value').textContent = e.target.value + ' Hz';
        this.updateSynthesisVisualization();
      });
    }
    
    if (generateBtn) {
      generateBtn.addEventListener('click', () => {
        this.generateAndPlayAudio();
      });
    }
    
    // Initial visualization
    this.updateSynthesisVisualization();
  }
  
  updateSynthesisVisualization() {
    const canvas = document.getElementById('synthesis-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const carrierFreq = parseFloat(document.getElementById('carrier-freq')?.value || 440);
    const beatFreq = parseFloat(document.getElementById('beat-freq')?.value || 100);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.drawGrid(ctx, canvas.width, canvas.height);
    
    // Generate preview waveform
    const previewData = this.generatePreviewWaveform(carrierFreq, beatFreq);
    this.drawWaveform(ctx, canvas.width, canvas.height, previewData);
  }
  
  generatePreviewWaveform(carrierFreq, beatFreq) {
    const length = 400;
    const data = new Array(length);
    const sampleRate = 8000; // samples per second
    const duration = 0.1; // 100ms preview
    
    for (let i = 0; i < length; i++) {
      const t = (i / length) * duration;
      const carrier = Math.sin(2 * Math.PI * carrierFreq * t * 0.01); // Scale down for visualization
      const beat = 0.5 * (1 + Math.sin(2 * Math.PI * beatFreq * t * 0.01));
      data[i] = carrier * beat;
    }
    
    return data;
  }
  
  generateAndPlayAudio() {
    if (!this.audioContext) {
      alert('Web Audio API not supported in this browser');
      return;
    }
    
    const carrierFreq = parseFloat(document.getElementById('carrier-freq')?.value || 440);
    const beatFreq = parseFloat(document.getElementById('beat-freq')?.value || 100);
    
    this.synthesizer.generateBeatNote(this.audioContext, carrierFreq, beatFreq);
  }
}

class AudioSynthesizer {
  generateBeatNote(audioContext, carrierFreq, beatFreq) {
    const duration = 3; // 3 seconds
    const sampleRate = audioContext.sampleRate;
    const length = sampleRate * duration;
    
    // Create audio buffer
    const buffer = audioContext.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);
    
    // Generate beat note
    for (let i = 0; i < length; i++) {
      const t = i / sampleRate;
      const carrier1 = Math.sin(2 * Math.PI * carrierFreq * t);
      const carrier2 = Math.sin(2 * Math.PI * (carrierFreq + beatFreq) * t);
      data[i] = (carrier1 + carrier2) * 0.3; // Reduce volume
    }
    
    // Play the generated audio
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start();
  }
}

// Global instance
let audioLab;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  audioLab = new AudioLaboratory();
});

// Expose methods for button callbacks
window.audioLab = {
  zoomIn: function(demoNumber) {
    console.log(`Zoom in demo ${demoNumber}`);
  },
  zoomOut: function(demoNumber) {
    console.log(`Zoom out demo ${demoNumber}`);
  },
  resetZoom: function(demoNumber) {
    console.log(`Reset zoom demo ${demoNumber}`);
  }
};