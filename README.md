# Fur player

Real-time player for a reconstructed dog coat: 600,000 fur strands drawn as 7.3 million strand-aligned 3D Gaussians,
with live fur dynamics, motion-captured motions and a stroking tool. It runs in the browser with WebGPU (Chrome or Edge
113 or later; a desktop GPU is recommended). The page loads about 180 MB of data.

To run it locally, serve this folder, for example `python -m http.server 8000`, and open http://localhost:8000/.

Controls: drag to orbit, right-drag to pan, wheel to zoom, double-click to reset the view, [H] to stroke the fur,
[Space] to play or pause, [1]-[6] for the motions. Settings are behind the sliders button at the bottom right.

Licences: the code is Apache-2.0; the Gaussian projection is ported from gsplat (Apache-2.0); the dog motion capture
is by Zhang, Starke, Komura and Saito (2018, AI4Animation), CC BY-NC 4.0, for non-commercial use only.
