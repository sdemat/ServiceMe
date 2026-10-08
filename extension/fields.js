// Form fields the auto-fill sets, found by their on-page label. Edit rows to match your form.
// kind: "reference" (lookup box), "select" (dropdown) or "text". source: key of the value to use (see popup.js).
// Fields not on the page are skipped and reported. Nothing here ever saves or submits the form.
export const FORM_FIELDS = [
  { label: "Requested For", kind: "reference", source: "eid" },
  { label: "Location", kind: "reference", source: "location" },
  { label: "Contact Method", kind: "select", source: "contactMethod" },
  { label: "Assignment Group", kind: "reference", source: "assignmentGroup" },
  { label: "Assigned To", kind: "reference", source: "assignedTo" },
  { label: "Ticket Type", kind: "select", source: "ticketType" },
  { label: "Short description", kind: "text", source: "shortDescription" },
];

export const CONTACT_METHOD = "Walk-In";                     // always picked
export const TICKET_TYPES = ["General Inquiry", "Incident"]; // Ticket Type options (the inquiry/incident guess)
