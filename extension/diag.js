// Describes the form structure of the open page (every frame) so the filler can be matched to it.
// Reads field names and layout only: never field values or ticket text. Self-contained.
export function inspectPage(names) {
  const norm = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
  const CONTROL = "input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]), select, textarea";
  const deep = (sel, root = document) => {
    const out = [...root.querySelectorAll(sel)];
    for (const el of root.querySelectorAll("*")) if (el.shadowRoot) out.push(...deep(sel, el.shadowRoot));
    return out;
  };
  const short = (e) => e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + (e.classList.length ? "." + [...e.classList].slice(0, 2).join(".") : "");
  const labels = deep("label");
  const controls = deep(CONTROL);
  const labelText = (l) => (l.querySelector(".label-text") || l).textContent.replace(/\s+/g, " ").trim();
  return {
    page: location.origin + location.pathname,
    isTop: window === window.top,
    labels: labels.length,
    controls: controls.length,
    iframes: document.querySelectorAll("iframe").length,
    shadowHosts: [...document.querySelectorAll("*")].filter((e) => e.shadowRoot).length,
    labelTexts: labels.slice(0, 50).map((l) => labelText(l).slice(0, 70)),
    wanted: names.map((n) => {
      const w = norm(n);
      const l = labels.find((x) => norm(labelText(x)) === w) || labels.find((x) => norm(labelText(x)).includes(w));
      const aria = controls.find((c) => norm(c.getAttribute("aria-label")) === w);
      const near = l && (l.htmlFor ? l.getRootNode().querySelector(`[id="${l.htmlFor}"], [id="sys_display.${l.htmlFor}"]`) : null);
      const chain = []; for (let e = l && l.parentElement, i = 0; e && i < 4; e = e.parentElement, i++) chain.push(short(e));
      return {
        name: n, labelFound: !!l, labelFor: l ? l.htmlFor : null, labelTag: l ? short(l) : null,
        controlByFor: near ? short(near) + ":" + (near.type || "") : null,
        controlByAria: aria ? short(aria) : null, parents: chain,
      };
    }),
  };
}
