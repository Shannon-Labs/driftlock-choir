# Patent Folder

This directory consolidates all patent-related materials for the Bown Driftlock Method (Chronometric Interferometry for wireless synchronization).

Contents

- PROVISIONAL_PATENT_APPLICATION.md — updated, comprehensive provisional application draft (2025) ready for filing.
- Claims_Bown_Driftlock_Method.md — extracted claims for quick review and iteration.
- Prior_Art_Search_Report.md — consolidated prior-art analysis and prosecution positioning.
- Drawings_Plan.md — figure descriptions and Mermaid source; includes a script to plot convergence.
- figures/ — Mermaid diagram sources and generated artifacts (where applicable).
- Assignment_and_Entity.md — assignee and inventor information stub.

Notes

- The root-level PROVISIONAL_PATENT_APPLICATION.md is left intact for backward compatibility but now points to this folder as the authoritative draft.
- Figures use Mermaid for portability; optional Python script generates convergence plot PNG/SVG.
- Use `scripts/export_patent_assets.sh` to build a filing packet under `dist/patent_packet` (includes spec.html/pdf, drawings.pdf, abstract, assignment stub, figures/).
