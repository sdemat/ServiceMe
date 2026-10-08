// Opens the new-contact page and fills it. Values live only in memory, for at most WAIT_MS.
import { fillForm } from "./fill.js";
import { FORM_FIELDS } from "./fields.js";

const WAIT_MS = 120000;

async function autofill(values) {
  const { contactUrl } = await chrome.storage.local.get("contactUrl");
  if (!contactUrl) return;
  const origin = new URL(contactUrl).origin;
  let tabId = null, busy = false, finished = false, poll = null;

  const finish = () => {
    finished = true;
    clearInterval(poll);
    values = null;
  };
  const tryFill = async () => {
    if (busy || finished) return;
    busy = true;
    try {
      const tab = await chrome.tabs.get(tabId);
      if (tab.url && tab.url.startsWith(origin)) { // skip sign-in pages on other sites
        const results = await chrome.scripting.executeScript({
          target: { tabId, allFrames: true }, func: fillForm, args: [values, FORM_FIELDS],
        });
        if (results.some((r) => r.result && r.result.found)) {
          chrome.storage.session.remove("draft"); // filled: the saved description is no longer needed
          finish();
        }
      }
    } catch (e) { /* page not ready or not permitted yet; try again */ }
    busy = false;
  };

  const tab = await chrome.tabs.create({ url: contactUrl });
  tabId = tab.id;
  poll = setInterval(tryFill, 1000); // forms render late; keep trying until WAIT_MS
  setTimeout(() => { if (!finished) finish(); }, WAIT_MS);
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "autofill") autofill(msg.values);
});
