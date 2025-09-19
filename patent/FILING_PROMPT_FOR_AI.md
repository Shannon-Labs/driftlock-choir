# Filing Helper Prompt (for your next AI / Codex)

Goal: Guide me through filing this provisional patent in USPTO Patent Center (EFS), validate the packet contents, and answer questions live.

Context to load:
- Provisional HTML: `patent/PROVISIONAL_PATENT_APPLICATION.html`
- Provisional PDF: `dist/patent_packet/spec.pdf`
- Drawings PDF: `dist/patent_packet/drawings.pdf`
- Abstract: `dist/patent_packet/abstract.txt`
- ADS stub: `dist/patent_packet/ADS_Stub.txt`
- Figures (backups): `dist/patent_packet/figures/`

Tasks for the AI:
1) Pre‑flight checklist
- Confirm all required artifacts exist and are recent: spec.pdf, drawings.pdf, abstract.txt.
- Confirm Abstract ≤150 words and matches spec’s abstract.
- Confirm embedded figure captions present and SVGs exported to PNG (in figures/).
- Confirm date references set to September 19, 2025 and inventor “Hunter Bown” present.

2) USPTO Patent Center filing steps (Provisional)
- Log in to Patent Center; create “Provisional Application”.
- Add application details; upload files in this order: specification (spec.pdf), drawings (drawings.pdf), abstract (abstract.txt).
- Add ADS info from ADS_Stub.txt; fill inventor/assignee addresses and correspondence.
- Review, calculate fees, pay, and submit; download and save receipts/acknowledgments.

3) Post‑filing actions
- Save the Filing Receipt + EFS Acknowledgment PDFs into `patent/filing/`.
- Create a `FILED_METADATA.json` with application number, submission date/time, filer, and attached documents.
- Prepare a task list for non‑provisional and PCT filing windows (12‑month deadline), and reminders.

4) Answering questions
- Be ready to explain differences between Spec/Abstract/Claims/Drawings and what belongs where.
- Provide short, clear answers for USPTO form questions (e.g., entity status, inventor declaration timing, assignment). Avoid legal advice; point to the official USPTO instructions if uncertain.

5) Safety / Caution
- Do not alter any claims or tech content during filing. We are only uploading the already‑prepared PDFs and text.
- If a mismatch is found (e.g., abstract >150 words), stop and provide exact edit instructions before proceeding.

Ready prompts I can paste to you:

- “Run the pre‑flight checklist on my packet and tell me what (if anything) needs attention before filing.”
- “Walk me through the Patent Center screens step by step; pause after each step to confirm.”
- “Generate the FILED_METADATA.json schema and a template for me to fill post‑submission.”
- “What should I keep for records after filing? Where should I upload them in the repo?”
- “What exact docs do I need for the non‑provisional within 12 months?”
