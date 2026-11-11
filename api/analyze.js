// Optional server-side recognition with Claude.
// Set ANTHROPIC_API_KEY in your host's environment variables to turn it on.
// Without it, GET returns {enabled:false} and the site uses on-device recognition.
const PROMPT = "You are the vision step of a waste-sorting app. Look at the photo and identify the single most prominent item someone wants to throw away.\n\nChoose \"item\" from exactly these keys:\n- plastic_bottle: PET/HDPE bottles: water, soda, shampoo, milk\n- glass_bottle: intact glass bottles and jars\n- metal_can: aluminium drink cans, tin food cans\n- cardboard: corrugated or carton board, pizza boxes, delivery boxes\n- paper: newspaper, office paper, magazines, paper bags\n- tissue: used tissues, paper napkins, paper towels\n- paper_cup: paper coffee/tea cups, plastic-lined cups\n- food_container: takeaway boxes, tubs, trays, yogurt cups\n- foil_tray: aluminium foil, foil takeaway trays\n- carton: Tetra Pak milk/juice cartons\n- plastic_bag: carry bags, cling film, bubble wrap\n- snack_wrapper: chips packets, chocolate and biscuit wrappers, multilayer pouches\n- styrofoam: thermocol, foam cups and trays\n- food_scrap: peels, leftovers, tea leaves, eggshells, flowers\n- battery: AA/AAA cells, button cells, power banks\n- bulb: CFL, tube lights, LED bulbs\n- medicine: expired tablets, blister strips, syrups\n- e_waste: phones, chargers, cables, earphones, small gadgets\n- paint_can: paint tins, pesticide or solvent containers\n- sanitary: diapers, sanitary pads, masks, used gloves\n- broken_glass: broken glasses, mirrors, crockery\n- textile: old clothes, shoes, bags, fabric\n- unknown: nothing that matches, or no clear item\n\nAlso judge food or drink residue ON the item (grease stains, sauce, leftover food, sticky liquid): \"none\", \"light\" or \"heavy\". Labels, printing and normal colours are not residue.\n\nDo not force a match. \"match\" says how well the item fits the chosen key: \"exact\" (it is that thing), \"close\" (same material and use, e.g. a shampoo bottle for plastic_bottle) or \"weak\" (only loosely similar: a plastic lighter is NOT a plastic_bottle, a ceramic mug is NOT a glass_bottle). Use \"unknown\" with match \"weak\" when nothing fits.\n\nAlways give your own best guess of the category, so the app can still help with items outside the list: one of recyclable, compostable, general, special (hazardous: batteries, chemicals, flammables, mercury, electronics, medicines, aerosols, sharps), sanitary (body-fluid waste), reuse (worth donating).\n\"hazard\": short warning if the item is sharp, flammable, pressurised, toxic, infectious or contains a battery; otherwise empty.\n\nReply with ONLY a JSON object, no prose:\n{\"item\": \"<key>\", \"match\": \"exact|close|weak\", \"label\": \"<specific 2-5 word name, e.g. 'Coca-Cola aluminium can'>\", \"confidence\": <0-1>, \"residue\": \"none|light|heavy\", \"residue_note\": \"<max 12 words, or empty>\", \"box\": [x1, y1, x2, y2], \"note\": \"<max 15 words: anything that changes disposal, e.g. 'cap is a different plastic', 'contains liquid', or empty>\", \"guess_category\": \"<category>\", \"guess_reason\": \"<max 20 words: why, and how to prepare it>\", \"hazard\": \"<max 12 words, or empty>\"}\n\"box\" is the item's bounding box as fractions of image width and height (0-1).";
const MODEL = process.env.CLAUDE_MODEL || "claude-haiku-5-5";

export default async function handler(req, res) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (req.method === "GET") return res.status(200).json({ enabled: !!key });
  if (req.method !== "POST") return res.status(405).json({ error: "method not allowed" });
  if (!key) return res.status(503).json({ error: "server recognition is not configured" });

  const image = req.body && req.body.image;
  if (typeof image !== "string" || image.length < 100 || image.length > 2_000_000 || !/^[A-Za-z0-9+/=]+$/.test(image)) {
    return res.status(400).json({ error: "send {image: <base64 JPEG>} under about 1.5 MB" });
  }

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 600,
        messages: [{ role: "user", content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: image } },
          { type: "text", text: PROMPT },
        ] }],
      }),
    });
    if (!r.ok) return res.status(r.status === 429 ? 429 : 502).json({ error: "upstream " + r.status });
    const data = await r.json();
    const text = (data.content || []).map(c => c.text || "").join("");
    const m = text.match(/\{[\s\S]*\}/);
    return res.status(200).json(JSON.parse(m ? m[0] : text));
  } catch (e) {
    return res.status(502).json({ error: "recognition failed" });
  }
}
