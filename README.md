# Waste Disposal Coach

Point a camera at an item and get told which bin it goes in, and how to prepare it.

**Live:** https://vatsalwadhi.github.io/waste-disposal/

![flow](https://img.shields.io/badge/camera%20→%20recognise%20→%20check%20residue%20→%20local%20rules-1f7a4d)

## What it does

- **Recognises** bottles, cans, cardboard, wrappers, cartons, food scraps, batteries, bulbs, medicines, e-waste, sanitary items and more (22 item types).
- **Classifies** each item: recyclable, compostable, general, special care, sanitary or reuse.
- **Checks for contamination** like grease on a pizza box, and highlights the stain on the photo. A greasy box goes to a different bin than a clean one.
- **Local rules decide the final answer**, not the AI. Profiles included: India (Solid Waste Management Rules 2026 four-stream: wet / dry / sanitary / special care), a generic three-bin system, and a city that composts soiled paper.
- **Items outside the list** get a clearly marked best guess, with safety warnings for sharp, flammable or toxic things.
- Live webcam, photo capture on phones, upload, drag-and-drop, built-in sample photos, recent-scan history.

## Recognition engines

The page picks the best one available:

| Engine | When | Cost |
|---|---|---|
| On-device AI (CLIP ViT-B/32 via transformers.js) | Default. Downloads ~90 MB once, then runs in the browser | Free, no key, photos never leave the device |
| Claude vision via `/api/analyze` | When `ANTHROPIC_API_KEY` is set on the host | Pay-per-use API |
| Claude vision via claude.ai | When the page is opened as a claude.ai artifact | Included |

Contamination uses two signals: the model's judgement, plus a colour-based stain map (k-means in LAB, then food-coloured pixels that differ from the base material).

## Deploy

**Vercel (recommended, supports the optional API):** import this repo at vercel.com/new. No build settings needed. To use Claude recognition, add `ANTHROPIC_API_KEY` under Settings → Environment Variables and redeploy.

**Render / GitHub Pages / Netlify:** deploy as a static site with the repo root as the publish directory. Recognition runs on-device.

Run locally: any static server, e.g. `npx serve .` (use `vercel dev` to include the API).

## Add your city

In `index.html`, copy a block in `REGIONS`, rename the bins and override only the items your municipality treats differently.

## Python prototype

`python/` holds the original desktop version: YOLO-World + OpenCV + a JSON rules file, with a live webcam window. See `python/README.md`.

## Credits

Sample photos from [TrashNet](https://github.com/garythung/trashnet) (Gary Thung & Mindy Yang, MIT licence). The pizza-box grease stains were added for the demo. Rules are guidance: your local body's published list always wins.
