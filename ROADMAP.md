# Technical Roadmap

This document outlines the development and research roadmap for the Driftlock Choir project. Our strategy is to transition from a proven simulation to a hardware-validated technology ready for commercial application.

## Phased Development Plan

### Phase 1: Foundational Credibility (Current Phase: In Progress)
*Goal: Prove the core algorithm works on real hardware in a controlled environment.*

*   [ ] **Recruit Technical Partner:** Secure a lead engineer with deep expertise in RF systems, DSP, and FPGA design.
*   [ ] **Procure Hardware:** Acquire two high-quality Software-Defined Radios (SDRs) and two high-stability Oven-Controlled Crystal Oscillators (OCXOs).
*   [ ] **Execute "Cabled" Experiment:** Validate the simulation results by replacing the simulated channel with a direct coaxial cable connection. This is the most critical next milestone to establish baseline hardware performance.
*   [X] **Strategic Repositioning:** Update repository documentation to reflect project ambition and strategy.

### Phase 2: Real-World Validation & Dissemination
*Goal: Demonstrate the technology in a realistic environment and establish scientific credibility.*

*   [ ] **Anechoic Chamber Testing:** Conduct over-the-air tests in a reflection-free environment to validate the 13.5 ps precision claim.
*   [ ] **Initial Publication:** Submit a paper to a peer-reviewed conference (e.g., IEEE ICC) based on cabled and anechoic results.
*   [ ] **Multipath Characterization:** Begin over-the-air tests in a real-world office/lab environment to gather data on multipath interference.

### Phase 3: Commercialization & Standardization
*Goal: Develop a commercially viable product and influence industry standards.*

*   [ ] **Develop "Performance Layer":** Begin implementation of estimators and control loops on an FPGA for real-time, high-performance operation.
*   [ ] **Develop Multipath Mitigation Algorithms:** Research and implement advanced estimators to overcome the challenges of real-world multipath environments.
*   [ ] **Engage Industry Partners:** Use hardware results to build formal collaborations with academic, semiconductor, and telecommunications partners.
*   [ ] **Engage with Standards Bodies:** Participate in relevant standards bodies (e.g., 3GPP, IEEE) to position Driftlock Choir as a foundational technology for future wireless systems.

## Current Research & Development Challenges

We are actively focused on solving the key challenges that separate simulation from a robust, real-world system:

1.  **Multipath & Fading:** In real environments, RF signals reflect off multiple surfaces, creating complex interference. Developing estimators that are robust to these effects is the primary research focus.
2.  **RF Front-End Stability:** Real-world amplifiers, mixers, and filters have phase and amplitude characteristics that drift with temperature and time. A robust calibration scheme is required to compensate for these hardware imperfections.
3.  **Antenna Phase Center:** The effective point of signal radiation from an antenna can vary with frequency and angle, introducing timing errors. This must be characterized and calibrated.
4.  **Real-Time Processing:** The Maximum Likelihood and other advanced estimators can be computationally intensive. Implementing these efficiently on an embedded platform (FPGA/DSP) is critical for a practical device.
