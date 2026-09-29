'use strict';
document.documentElement.classList.add('js');

const hero = document.querySelector('.hero-video');
const heroToggle = document.querySelector('.hero-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let heroEnabled = !reducedMotion.matches;
let heroVisible = true;
const updateHeroLabel = () => {
  heroToggle.textContent = hero.paused ? 'Play background' : 'Pause background';
};
const playHero = () => {
  if (!hero.getAttribute('src')) hero.src = hero.dataset.src;
  hero.play().catch(updateHeroLabel);
};
heroToggle.hidden = false;
heroToggle.addEventListener('click', () => {
  heroEnabled = hero.paused;
  if (heroEnabled) playHero(); else hero.pause();
});
hero.addEventListener('play', updateHeroLabel);
hero.addEventListener('pause', updateHeroLabel);
new IntersectionObserver(([entry]) => {
  heroVisible = entry.isIntersecting;
  if (heroVisible && heroEnabled && !document.hidden) playHero(); else hero.pause();
}, {threshold: 0.05}).observe(hero);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && heroVisible && heroEnabled) playHero(); else hero.pause();
});
reducedMotion.addEventListener('change', event => {
  heroEnabled = !event.matches;
  if (heroEnabled && heroVisible) playHero(); else hero.pause();
});
updateHeroLabel();

// Native controls remain usable without JavaScript. Load metadata only near the viewport.
const videoObserver = new IntersectionObserver(entries => {
  entries.forEach(({target, isIntersecting}) => {
    if (isIntersecting && target.preload === 'none') target.preload = 'metadata';
    if (!isIntersecting) target.pause();
  });
}, {rootMargin: '150px'});
document.querySelectorAll('video:not(.hero-video)').forEach(video => {
  videoObserver.observe(video);
  video.addEventListener('play', () => {
    document.querySelectorAll('video:not(.hero-video)').forEach(other => {
      if (other !== video) other.pause();
    });
  });
});

const selector = document.querySelector('.example-selector');
selector.hidden = false;
const exampleButtons = [...selector.querySelectorAll('button')];
function selectExample(button) {
  exampleButtons.forEach(item => {
    const active = item === button;
    item.setAttribute('aria-pressed', String(active));
    const panel = document.getElementById(item.getAttribute('aria-controls'));
    panel.hidden = !active;
    if (!active) panel.querySelector('video').pause();
  });
}
exampleButtons.forEach(button => button.addEventListener('click', () => selectExample(button)));
selectExample(exampleButtons[0]);

// Policy clips are grouped by terrain; native selects also support keyboard use.
const motionControls = document.querySelector('.motion-controls');
const terrainSelect = document.getElementById('terrain-category');
const motionSelect = document.getElementById('policy-motion');
const policyPanels = [...document.querySelectorAll('.policy-panel')];
const rememberedMotions = new Map();
const policyStatus = document.getElementById('policy-selection-status');

function showPolicyMotion() {
  policyPanels.forEach(panel => {
    const active = panel.id === motionSelect.value;
    panel.hidden = !active;
    if (!active) panel.querySelector('video').pause();
  });
  const selected = document.getElementById(motionSelect.value);
  rememberedMotions.set(terrainSelect.value, motionSelect.value);
  policyStatus.textContent = selected.dataset.label + ' selected.';
}

function chooseTerrain() {
  const matching = policyPanels.filter(panel => panel.dataset.terrain === terrainSelect.value);
  motionSelect.replaceChildren(...matching.map(panel => new Option(panel.dataset.label, panel.id)));
  const previous = rememberedMotions.get(terrainSelect.value);
  if (matching.some(panel => panel.id === previous)) motionSelect.value = previous;
  showPolicyMotion();
}

terrainSelect.addEventListener('change', chooseTerrain);
motionSelect.addEventListener('change', showPolicyMotion);
chooseTerrain();
motionControls.hidden = false;

const copyButton = document.querySelector('.copy-button');
const copyStatus = document.querySelector('.copy-status');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
