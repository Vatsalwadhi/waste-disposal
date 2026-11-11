# Held-out evaluation set

`heldout.json`: 40 TrashNet photos (8 each of cardboard, glass, metal, paper, plastic),
picked with `random.seed(42)` and excluding the 13 built-in sample photos the
on-device labels were tuned on. Images are 224 px JPEG data URLs.

Run it: open the live site, then in the browser console:

```js
const set = await (await fetch("eval/heldout.json")).json();
const MAT = {cardboard:["cardboard"], glass:["glass_bottle","broken_glass"], metal:["metal_can","foil_tray","paint_can"],
             paper:["paper","carton"], plastic:["plastic_bottle","food_container","plastic_bag","carton","styrofoam"]};
let ok = 0; const rows = [];
for (const t of set) {
  const im = new Image(); im.src = t.src; await im.decode();
  const c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight;
  c.getContext("2d").drawImage(im, 0, 0); baseImage = c;
  const r = await clipAnalyse(); const hit = MAT[t.cls].includes(r.item); ok += hit;
  rows.push([t.cls, t.file, r.item, r.confidence.toFixed(2), hit]);
}
console.table(rows); console.log(ok + "/" + set.length);
```

Photos: TrashNet, Gary Thung and Mindy Yang, MIT licence.
