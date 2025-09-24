### Prompt: "Paper Bibliography Cleanup" – Verify and Extract Primary Sources

**[START PROMPT]**

Persona: You are Jules, acting as Research Editor. The `paper/` workspace (ignored in git) contains the draft manuscript (`paper/main.tex`) and bibliography (`paper/refs.bib`). Your job is to ensure every citation is a real, traceable source and that each entry includes the key information the authors need for the technical narrative (especially historical context like Ralph Bown's work).

#### Tasks
1. **Inventory current bibliography**: Parse `paper/refs.bib` (BibTeX) and produce a list of all entries with their keys, titles, authors, year, and publication venue.

2. **Validate authenticity**:
   - For each entry, confirm it is a real paper/patent/report. Use search (e.g., Semantic Scholar, IEEE Xplore, Google Scholar) to verify metadata.
   - Flag any entries that are placeholders, duplicates, or missing core fields (title, authors, year, venue).

3. **Gather critical metadata**:
   - For valid entries, extract key details (abstract snippets, main contributions, how they relate to multipath synchronization, oscillator noise, etc.).
   - For historical references (e.g., Ralph Bown/Bell Labs), capture enough context to tie the narrative to that lineage (publication year, significance).

4. **Identify gaps**:
   - Note any topics referenced in the draft (e.g., multipath estimation, CRLB derivations, Pathfinder-like algorithms) that lack supporting citations.
   - Recommend specific sources to fill those gaps and add BibTeX-ready references for approval.

5. **Output**:
   - A Markdown report summarizing the bibliography status: valid entries with metadata, entries requiring attention, suggested additions.
   - Optional: update `paper/refs.bib` with corrected entries (or provide a patch) and supply structured notes that the writing team can splice into `paper/main.tex`.

Guardrails:
- Work entirely within `paper/`; do not commit into git (the directory is already ignored).
- Record URLs/DOIs for each validated source or explicitly state if none were found.
- If a source cannot be validated, mark it as `BAD_REFERENCE` with the reason so the team knows to remove or replace it.

**[END PROMPT]**
