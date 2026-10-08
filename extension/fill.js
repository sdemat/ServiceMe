// Runs inside the ServiceNow page (every frame). Self-contained: Chrome serializes this function.
// Sets values only. It never clicks, submits or saves.
export async function fillForm(values, fields) {
  const norm = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
  const CONTROL = "input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]), select, textarea";
  // querySelectorAll that also looks inside shadow roots
  const deep = (sel, root = document) => {
    const out = [...root.querySelectorAll(sel)];
    for (const el of root.querySelectorAll("*")) if (el.shadowRoot) out.push(...deep(sel, el.shadowRoot));
    return out;
  };
  const labelText = (l) => norm((l.querySelector(".label-text") || l).textContent);
  const controlFor = (label) => {
    if (label.htmlFor) {
      const root = label.getRootNode();
      const byId = (id) => root.querySelector(`[id="${id}"]`);
      const el = byId("sys_display." + label.htmlFor) || byId(label.htmlFor);
      if (el && el.type !== "hidden") return el;
    }
    let box = label.parentElement;
    for (let d = 0; box && d < 4; d++, box = box.parentElement) {
      const el = [...box.querySelectorAll(CONTROL)].find((c) => label.compareDocumentPosition(c) & Node.DOCUMENT_POSITION_FOLLOWING);
      if (el) return el;
    }
    return null;
  };
  const locate = (name) => {
    const want = norm(name);
    const labels = deep("label").map((l) => [l, labelText(l)]);
    const part = labels.filter(([, t]) => t.includes(want));
    const hit = labels.find(([, t]) => t === want) || (part.length === 1 ? part[0] : null);
    const viaLabel = hit && controlFor(hit[0]);
    if (viaLabel) return viaLabel;
    const all = deep(CONTROL);
    return all.find((c) => norm(c.getAttribute("aria-label")) === want) || all.find((c) => norm(c.getAttribute("placeholder")) === want) || null;
  };
  const setValue = (el, v) => {
    const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, v);
  };
  const fire = (el, ...types) => types.forEach((t) => el.dispatchEvent(new Event(t, { bubbles: true })));
  const pickOption = (sel, want) => {
    const w = norm(want), opts = [...sel.options];
    return opts.find((o) => norm(o.text) === w) || opts.find((o) => norm(o.value) === w) ||
      (w ? opts.find((o) => norm(o.text).startsWith(w)) : null);
  };

  const wanted = fields.filter((f) => values[f.source]);
  const deadline = Date.now() + 3000;
  while (!wanted.some((f) => locate(f.label)) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 250));
  if (!wanted.some((f) => locate(f.label))) return { found: false };

  const report = [];
  for (const f of wanted) {
    const el = locate(f.label);
    if (!el) { report.push([f.label, "not on this page"]); continue; }
    if (el.disabled || el.readOnly) { report.push([f.label, "locked"]); continue; }
    const v = String(values[f.source]);
    if (f.kind === "select") {
      const opt = el.tagName === "SELECT" && pickOption(el, v);
      if (!opt) { report.push([f.label, "no matching option"]); continue; }
      el.value = opt.value;
      fire(el, "input", "change");
      report.push([f.label, "filled"]);
    } else if (f.kind === "reference") {
      el.focus();
      for (let i = 1; i <= v.length; i++) { // type the whole string, one key at a time, so the lookup sees every word
        setValue(el, v.slice(0, i));
        fire(el, "input");
        const key = v[i - 1];
        el.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key }));
        el.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key }));
      }
      await new Promise((r) => setTimeout(r, 800)); // let the lookup run, then leave the field like Tab would
      el.blur();
      fire(el, "change");
      // wait until ServiceNow recognizes the entry (its hidden id value is set, or the text turns into the record's name)
      const hid = el.id.startsWith("sys_display.") && el.getRootNode().querySelector(`[id="${el.id.slice(12)}"]`);
      const done = () => (hid ? hid.value : el.value !== v);
      for (let i = 0; i < 40 && !done(); i++) await new Promise((r) => setTimeout(r, 250));
      report.push([f.label, done() ? "recognized" : "typed, but not recognized; check it"]);
    } else {
      setValue(el, v);
      fire(el, "input", "change");
      report.push([f.label, "filled"]);
    }
  }

  const box = document.createElement("div");
  box.style.cssText = "position:fixed;top:12px;right:12px;z-index:2147483647;max-width:340px;padding:10px 12px;" +
    "background:#fffbe6;border:1px solid #c9a300;border-radius:6px;font:13px/1.4 sans-serif;color:#222;box-shadow:0 2px 8px rgba(0,0,0,.25)";
  const head = document.createElement("div");
  head.style.cssText = "font-weight:600;margin-bottom:4px";
  head.textContent = "Service Me filled this form. Review it, then save it yourself.";
  box.append(head);
  for (const [name, status] of report) {
    const line = document.createElement("div");
    line.textContent = `${name}: ${status}`;
    box.append(line);
  }
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.style.cssText = "margin-top:6px";
  close.addEventListener("click", () => box.remove());
  box.append(close);
  document.body.append(box);
  return { found: true, report };
}
