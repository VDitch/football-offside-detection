# Automated Football Offside Detection

## Project Overview

This repository contains the diploma thesis project setup for **automated football offside detection** using computer vision. The goal is to detect players and the ball, estimate pitch geometry, normalize perspective, and support offside decision analysis from match video frames.

## Pipeline Architecture

The planned end-to-end pipeline is organized into the following stages:

1. **YOLO Player and Ball Detection**  
   Use YOLO models to detect all players and the ball in each frame.

2. **Team Classification**  
   Classify detected players into teams (and optionally referees/goalkeepers) based on visual features such as dominant kit colors.

3. **Pitch Line Detection with Hough Transform**  
   Detect key field markings using classical computer vision (Canny + Hough line detection) to estimate pitch orientation and structure.

4. **Homography Perspective Transformation**  
   Compute a homography from image coordinates to a canonical top-down pitch view for geometric consistency and more reliable offside-line reasoning.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VDitch/football-offside-detection.git
   cd football-offside-detection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\\Scripts\\activate   # Windows (PowerShell)
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Current repository state focuses on thesis project scaffolding:

- `src/` — source code modules
- `data/` — datasets and video assets (kept out of version control)
- `models/` — trained model files and checkpoints
- `results/` — generated outputs and experiment artifacts
- `docs/` — thesis documentation and project notes

As implementation progresses, processing scripts and notebooks should be added under `src/` and documented here.

## Visual Results

This section will host qualitative and quantitative outputs from experiments, for example:

- frame-level detections (players/ball)
- team-classification overlays
- detected pitch lines and vanishing geometry
- homography-transformed bird’s-eye visualizations
- final offside-line decision snapshots

Recommended location for generated figures: `results/visualizations/`.
