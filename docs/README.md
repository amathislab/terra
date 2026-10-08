# TERRA documentation and project website

## Use the code

Start with the [repository README](../README.md) for the first retargeted motion.
Before running code, [install TERRA](installation.md) and
[obtain data and models](data.md). Then follow the four stages of the method:

1. [Load or transform motion files](motion-files.md).
2. [Reconstruct terrain](terrain-reconstruction.md).
3. [Retarget a motion](motion-retargeting.md).
4. [Prepare a cohort and train a policy](policy-training.md).

For non-AMASS recordings, follow [Download a dataset and play a pre-trained policy](dataset-workflows.md#non-amass-dataset-to-a-pre-trained-policy).

## Maintain the project website

Static HTML, CSS, and JavaScript; no build step or external runtime dependencies.

Preview from the repository root:

```bash
python -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open http://localhost:8000. Edit `index.html` for text, videos, and the citation;
`styles.css` for appearance; and `script.js` for playback and example selection.
Fonts and media are self-hosted in `assets/`. Videos are silent H.264 MP4 files.
Policy videos and posters are in `assets/videos/policy/` and
`assets/images/policy/`. Each `.policy-panel` in `index.html` supplies the
category and motion label used by the selectors.

Publish the website through GitHub Pages from the public repository's
**main** branch and **/docs** folder. The lab maintains the
https://cnai.epfl.ch/terra/ URL separately; keep that mapping pointed at the
Pages site when changing repository settings.

The [arXiv preprint](https://arxiv.org/abs/2609.38653) is linked from the site.
Charter's license is included in `assets/fonts/LICENSE.txt`.
