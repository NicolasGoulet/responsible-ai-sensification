# Feature Backlog

Each item below is a self-contained feature for this branch, to be implemented in order.

## 1. Live waveform — Web Audio API AnalyserNode visualization
Replace the canvas mock-waveform (drawn from notes data) with a real-time audio waveform using `AnalyserNode.getFloatTimeDomainData()` connected to the Web Audio API graph built in Feature 1.

## 2. Image generation placeholder → actual image
The bottom-right panel is currently a static placeholder.
- Generate an image that encodes the SAE feature activations visually (e.g. scatter plot of active features by cluster, heatmap of activations over tokens, or a t-SNE/UMAP projection).
- Could also call an external image generation API conditioned on the generated text.
- Decide approach when implementing.

## 3. Neuronpedia download progress
The initial model + SAE + Neuronpedia load can take 30–60 s. Add a proper progress bar in the UI fed by server-sent `loading` events, showing each stage (model load, SAE load, Neuronpedia cache hit/download).

## 4. Session history & replay
- Save each completed run (prompt + all token events) to `runs/` as NDJSON.
- Add a "History" panel in the UI to list past runs and replay them without re-running the model.
- Reuse the existing `export.py` / `runs/` convention.

## 5. Instrument attribution per cluster 
- in a scroll box, display for each cluster:
  - the number of the cluster
  - which sound pack it plays (not clear how to do that at all, must be defined but the idea is to select which instrument is playing what is represented by this cluster)
  - the names of the features that were activated that are part of this cluster (to get a sense of what it represents)
