import { prepare, predict, bucketText, hoursText } from "./engine.js";
import { CONTACT_METHOD, FORM_FIELDS, TICKET_TYPES } from "./fields.js";
import { inspectPage } from "./diag.js";

const $ = (id) => document.getElementById(id);
const NONE = "";
let model = null, current = null, timer = null;

const FIELD_TITLES = { priority: "Priority", category: "Category", record_source: "Record source" };

async function init() {
  try {
    const blob = await (await fetch(chrome.runtime.getURL("model/model.json"))).json();
    model = prepare(blob);
    $("status").textContent = `${blob.tickets.ids.length} past tickets loaded.`;
  } catch (e) {
    $("status").textContent = "Model not found: run python src/export.py and copy extension/model/model.json here.";
  }
  const { draft } = await chrome.storage.session.get("draft"); // survives closing the popup; cleared when the browser closes
  if (draft) { $("desc").value = draft; analyze(); }
  const { contactUrl, location, group, assignee } = await chrome.storage.local.get(["contactUrl", "location", "group", "assignee"]);
  $("url").value = contactUrl || "";
  $("location").value = location || "";
  $("group").value = group || "";
  $("assignee").value = assignee || "";
  $("settings").open = !contactUrl;
  $("fill").disabled = !contactUrl;
}

// Mask EID, email and phone patterns so they never reach the model.
function scrub(text) {
  const c = model.cfg;
  return ["EID_PATTERN", "EMAIL_PATTERN", "PHONE_PATTERN"].reduce((t, k) => t.replace(new RegExp(c[k], "g"), " "), text);
}

function render(r) {
  const c = model.cfg, t = r.time;
  $("result").hidden = false;
  $("verdict").textContent = r.inquiry ? `${c.AT_THE_WINDOW_LABEL}` : `~${hoursText(t.median)}`;
  const range = `typically ${hoursText(t.low)} to ${hoursText(t.high)}, best case ${hoursText(t.best)}`;
  const rough = t.rough ? " (rough: few close matches)" : ` (${t.matches} similar incidents)`;
  $("time").textContent = r.inquiry
    ? `Looks like a general inquiry. If it is an incident: ~${hoursText(t.median)}, ${range}${rough}`
    : `Usually ${bucketText(t.median, c.BUCKET_EDGES)}; ${range}${rough}`;

  // Ticket Type comes from the inquiry/incident guess
  const pre = r.inquiry ? TICKET_TYPES[0] : TICKET_TYPES[1];
  const sel = $("ticket-type");
  sel.replaceChildren(...TICKET_TYPES.map((t) => new Option(t, t, t === pre, t === pre)), new Option("(don't fill)", NONE));
  $("type-hint").textContent = `Chance it is a general inquiry: ${Math.round(r.p_inquiry * 100)}%`;

  const info = $("info");
  info.replaceChildren();
  for (const [name, guesses] of Object.entries(r.fields)) {
    const row = document.createElement("div");
    row.className = "row";
    const lab = document.createElement("span");
    lab.textContent = FIELD_TITLES[name] || name;
    const val = document.createElement("span");
    val.textContent = guesses.length > 1 ? guesses.map(([v, p]) => `${v} (${Math.round(p * 100)}%)`).join(", ") : `${guesses[0][0]} (default)`;
    row.append(lab, val);
    info.append(row);
  }

  $("suggest-box").hidden = !r.suggestion;
  $("no-suggest").hidden = !!r.suggestion;
  $("use-suggest").checked = false;
  if (r.suggestion) $("suggest-text").textContent = r.suggestion;

  const list = $("similar");
  list.replaceChildren();
  for (const s of r.similar) {
    const li = document.createElement("li");
    const id = document.createElement("b");
    id.textContent = s.id;
    li.append(id, ` (${Math.round(s.sim * 100)}%) ${s.text.slice(0, 120)}`);
    list.append(li);
  }
}

function analyze() {
  const text = $("desc").value.trim();
  if (!model || text.length < 3) { $("result").hidden = true; current = null; return; }
  current = predict(model, scrub(text));
  render(current);
}

$("desc").addEventListener("input", () => {
  chrome.storage.session.set({ draft: $("desc").value }); // the description only; the EID is never saved
  clearTimeout(timer);
  timer = setTimeout(analyze, 300);
});

$("clear").addEventListener("click", () => {
  $("desc").value = "";
  $("eid").value = "";
  chrome.storage.session.remove("draft");
  analyze();
});

$("fill").addEventListener("click", () => {
  if (!current) return;
  const values = { contactMethod: CONTACT_METHOD, eid: $("eid").value.trim(),
    location: $("location").value.trim(), assignmentGroup: $("group").value.trim(), assignedTo: $("assignee").value.trim() };
  values.shortDescription = $("use-suggest").checked && current.suggestion ? current.suggestion : $("desc").value.trim();
  values.ticketType = $("ticket-type").value;
  chrome.runtime.sendMessage({ type: "autofill", values });
  $("eid").value = ""; // never kept
  $("status").textContent = "Opening the new contact page...";
});

$("save-settings").addEventListener("click", async () => {
  const msg = $("url-msg");
  try {
    const u = new URL($("url").value.trim());
    if (u.protocol !== "https:") throw new Error("Use an https:// address.");
    const origin = `${u.origin}/*`;
    if (!(await chrome.permissions.contains({ origins: [origin] })) && !(await chrome.permissions.request({ origins: [origin] }))) {
      throw new Error("Permission to fill that site was not granted.");
    }
    await chrome.storage.local.set({ contactUrl: u.href, location: $("location").value.trim(), group: $("group").value.trim(), assignee: $("assignee").value.trim() });
    $("fill").disabled = false;
    msg.textContent = "Saved.";
  } catch (e) {
    msg.textContent = e.message || "Not a valid URL.";
  }
});

$("check").addEventListener("click", async () => {
  const out = $("report");
  out.hidden = false;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const res = await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, func: inspectPage, args: [FORM_FIELDS.map((f) => f.label)] });
    const frames = res.map((r) => r.result).filter((r) => r && (r.labels || r.iframes || r.controls));
    out.value = JSON.stringify({ frames: frames.length ? frames : "no form fields found in any frame" }, null, 1);
  } catch (e) {
    out.value = "Could not read this page: " + e.message + "\n(Save the contact page URL in Settings first so Chrome allows access to that site.)";
  }
  out.select();
});

init();
