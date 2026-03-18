# NanoGS: Training-Free Gaussian Splat Simplification

<p align="center">
  <strong>Simplify your GS in the browser</strong>
</p>

<p align="center">
  <!-- arXiv -->
  <a href="https://arxiv.org/abs/2603.16103">
    <img src="https://img.shields.io/badge/arXiv-2603.16103-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <!-- Paper PDF -->
  <a href="https://arxiv.org/pdf/2603.16103">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF">
  </a>
  <!-- Project page / Web app -->
  <a href="https://rongliu-leo.github.io/NanoGS/">
    <img src="https://img.shields.io/badge/Project-Web%20Demo-1abc9c?style=for-the-badge&logo=googlechrome" alt="Project page">
  </a>
  <!-- Code (Python) -->
  <a href="https://github.com/saliteta/NanoGS">
    <img src="https://img.shields.io/badge/Code-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Code (Python)">
  </a>
</p>

---


<p align="center">
  <img src="teaser.png" alt="NanoGS teaser" width="80%">
</p>

<p align="center">
  <em>NanoGS achieves substantial simplification ratios while preserving fidelity and geometry—without post-optimization or calibrated images.</em>
</p>


## This repository (Web App)

It lets you:

- **Load** your own `.ply` Gaussian splat model or use built-in examples  
- **Simplify** with configurable target ratio and merge-cap parameters  
- **Compare** original vs. simplified splats interactively (trackball controls)  
- **Scrub** a progress track to inspect intermediate simplification stages

## How to use

Simply open the **Web App** [`https://rongliu-leo.github.io/NanoGS/`](https://rongliu-leo.github.io/NanoGS/) in a modern browser, load or pick a scene, and have fun simplifying and exploring your 3D Gaussian splats.

## Contribution

Your PRs are very welcome:

1. **Share your favorite scenes**: Add your own `.ply` Gaussian splat models under the `examples/` folder so others can explore and appreciate them in the web app.  
2. **Improve the implementation**: Help optimize the current codebase, especially around known limitations.

## Dependencies

The app uses **Three.js** for the scene/camera, **Spark** ([@sparkjsdev/spark](https://www.npmjs.com/package/@sparkjsdev/spark)) for splat rendering, and **static-kdtree** (`static-kdtree`, loaded via `https://esm.sh/static-kdtree`) for efficient KD-tree spatial queries; simplification runs in Web Workers where supported.

## Limitations

1. **Scene size**: Due to the underlying KD-tree JavaScript library, the web app may not handle very large Gaussian splat scenes reliably (e.g., \(>\) 5M splats). For large-scale scenes, please refer to the **Python implementation** instead: [`https://github.com/saliteta/NanoGS`](https://github.com/saliteta/NanoGS).  
2. **Very thin Gaussians**: Spark’s internal quantization can struggle to faithfully render extremely thin Gaussians. The **simplification algorithm itself still works correctly**, but if you want better visual quality for such cases, consider viewing the output with [Super Splat](`https://superspl.at/editor`) or another high-fidelity splat renderer.


## License

See `LICENSE` in this repository.
