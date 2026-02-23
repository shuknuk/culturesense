# CultureSense Demo Video Recording Checklist

## Pre-Recording Verification (All Complete)

- [x] All unit tests pass (36 tests)
- [x] Evaluation suite passes (24 checks)
- [x] Notebook builds successfully (31 cells)
- [x] Sample PDFs exist in `sampleReports/`:
  - `QuestDx_Report1_Week1_ActiveUTI.pdf`
  - `QuestDx_Report2_Week2_MidTreatment.pdf`
  - `QuestDx_Report3_Week3_Resolution.pdf`

---

## Google Colab Setup (Do This First)

### Step 1: Upload Notebook
1. Open [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook → Select `culturesense.ipynb`
3. Runtime → Change runtime type → **T4 GPU**

### Step 2: Run All Cells
1. Runtime → Run all
2. Wait ~3-5 minutes for MedGemma model to load
3. Look for: `MedGemma loaded on GPU.` or `Running with stub fallback`
4. Wait for Gradio to show: `Running on public URL: https://xxxxx.gradio.live`

### Step 3: Open Public URL
1. Click the `.gradio.live` link (opens in new tab)
2. **This is the URL you'll record from** (not the Colab cell output)

### Step 4: Test the Flow Once
1. Go to "Upload PDF" tab
2. Upload all 3 sample PDFs:
   - `QuestDx_Report1_Week1_ActiveUTI.pdf` (Week 1: 150,000 CFU)
   - `QuestDx_Report2_Week2_MidTreatment.pdf` (Week 2: 45,000 CFU)
   - `QuestDx_Report3_Week3_Resolution.pdf` (Week 3: 3,000 CFU)
3. Click "Process PDFs" → Wait for extraction
4. Review the table → Click "Confirm & Analyse"
5. Verify:
   - [ ] Patient output shows "Resolution Detected" (green)
   - [ ] Clinician output shows confidence ~90%
   - [ ] NO stewardship alert (infection cleared)
   - [ ] Resistance timeline shows "No high-risk resistance markers detected"

---

## Recording Session

### Technical Setup
- [ ] Browser: Full-screen or clean window (hide bookmarks bar)
- [ ] Resolution: 1080p minimum
- [ ] Screen recording: OBS, Loom, QuickTime, or browser extension
- [ ] Microphone: Enabled for live narration

### Narration Script (2 minutes)

> "Hello, we are Kinshuk Goel and Amit Goel.
>
> CultureSense is an AI-powered reasoning assistant designed specifically for urine and stool culture reports.
>
> Microbiology reports are complex. Clinicians must interpret organism identification, susceptibility tables, and evolving resistance patterns across multiple encounters. At the same time, patients often struggle to understand these reports, leading to confusion about antibiotic decisions.
>
> CultureSense addresses both challenges.
>
> In Clinician Mode, the system parses structured culture data, compares sequential reports, detects resistance evolution, and generates ranked clinical hypotheses with antimicrobial stewardship alerts.
>
> In Patient Mode, the same report is translated into simple, empathetic language. The system does not diagnose or prescribe. Instead, it explains what the findings mean and generates structured questions patients can discuss with their doctor.
>
> This reduces misunderstanding and supports informed conversations.
>
> Our hybrid approach uses prompt-engineered MedGemma combined with deterministic extraction and guardrails to ensure transparent, structured outputs.
>
> CultureSense enhances interpretability, antimicrobial stewardship awareness, and patient communication — while preserving clinical authority.
>
> Thank you."

### Visual Flow (Match to Narration)

| Time | Narration | Screen Action |
|------|-----------|---------------|
| 0:00 | "Hello, we are..." | Title card or home screen |
| 0:15 | "Microbiology reports are complex..." | Show sample Quest Diagnostics PDF |
| 0:30 | "Clinician Mode..." | Show Clinician output panel |
| 0:45 | "Patient Mode..." | Show Patient output panel |
| 1:00 | "Upload 3 PDFs..." | Upload the 3 sample PDFs |
| 1:15 | "Processing..." | Show "Process PDFs" loading |
| 1:25 | "Review extracted data..." | Show editable table |
| 1:35 | "Confirm & Analyse..." | Show progress bar stages |
| 1:45 | "Patient output..." | Show resolution alert (green) |
| 1:55 | "Clinician output..." | Show confidence badge, timeline |
| 2:10 | "Thank you." | Return to title/end card |

---

## Alternative: Resistance Alert Scenario

If you want to show the stewardship alert (red), you can use these values in the Manual Entry tab:

```
Date: 2026-01-01
Specimen: urine
Organism: Klebsiella pneumoniae
CFU/mL: 90000
Resistance Markers: (none)

Date: 2026-01-10
Specimen: urine
Organism: Klebsiella pneumoniae
CFU/mL: 80000
Resistance Markers: (none)

Date: 2026-01-20
Specimen: urine
Organism: Klebsiella pneumoniae
CFU/mL: 75000
Resistance Markers: ESBL
```

This will trigger:
- Stewardship Alert (red box)
- EMERGING_RESISTANCE flag
- Resistance timeline with ESBL marker

---

## Post-Recording

- [ ] Verify video is under 3 minutes
- [ ] Check audio clarity
- [ ] Ensure all UI elements are visible
- [ ] Confirm disclaimers visible in output
- [ ] Upload to YouTube (unlisted or public)
- [ ] Copy link for Kaggle Writeup

---

## Expected Output Verification

### Resolution Scenario (3 PDFs)
- Patient: Green "Resolution Detected" alert
- Clinician: Confidence ~90%, NO stewardship alert
- Resistance timeline: "No high-risk resistance markers detected"
- Patient questions: All 5 present

### Key Things to Show
1. **PDF upload flow** — drag and drop
2. **Processing indicator** — "Docling is extracting text"
3. **Editable table** — dates, organisms, CFU values
4. **Progress bar** — stages during analysis
5. **Green alert** — "Resolution Detected"
6. **Confidence badge** — "90%"
7. **Disclaimers** — visible at bottom of outputs

---

## Files to Bring to Colab

1. `culturesense.ipynb` — the notebook
2. `sampleReports/QuestDx_Report1_Week1_ActiveUTI.pdf`
3. `sampleReports/QuestDx_Report2_Week2_MidTreatment.pdf`
4. `sampleReports/QuestDx_Report3_Week3_Resolution.pdf`

---

## Troubleshooting

### Gradio URL not showing?
- Make sure cell with `demo.launch(share=True)` ran
- Look for output like: `Running on public URL: https://xxxxx.gradio.live`

### Model not loading?
- Verify GPU is enabled: Runtime → Change runtime type → T4 GPU
- Check for CUDA errors in cell output

### PDFs not extracting?
- Check debug output in the UI
- Verify Docling installed: first cell should install it

### Outputs look wrong?
- Re-run `build_notebook.py` locally
- Re-upload the notebook to Colab
- Run all cells again