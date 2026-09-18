import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Pap Smear Cytology Algorithm Project\nComprehensive Status, Audit, and Technical Roadmap")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Deep Navy
    title.paragraph_format.space_after = Pt(4)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("A Complete Plain-Language Review of Empirical Milestones, Experimental Reproducibility, Security Audit & Next Steps")
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    sub.paragraph_format.space_after = Pt(18)

    # Metadata Box
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Checkout", "C:\\Users\\abusu\\Desktop\\pap-smear-cyto-algo"),
        ("Date of Audit & Report", "September 14, 2026"),
        ("Primary Hardware Profile", "NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)"),
        ("Current Project Stage", "Phase 3 Completed / Dense Ground-Truth & Multi-Source Expansion")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.text = k
        c1.text = v
        c0.paragraphs[0].runs[0].font.bold = True
        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "FAFAFA")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Helper for headings
    def add_h1(text):
        h = doc.add_paragraph()
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2B, 0x54, 0x7E)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
        return h

    # Section 1: Executive Summary
    add_h1("1. Executive Plain-Language Summary")
    p = doc.add_paragraph(
        "This report consolidates all scientific findings, algorithm benchmarks, clinical data quirks, "
        "and security assessments for the Pap Smear Cytology detection project. "
        "Over the course of Phases 1 through 3 and subsequent deep-dive experiments, the algorithm transitioned "
        "from a single-dataset prototype to a robust multi-source cell detector. "
        "The most crucial discovery made during empirical auditing is that the primary dataset (SIPaKMeD) "
        "contains sparse annotations: human labelers previously marked only ~4 out of every 50+ visible cells per image. "
        "Recognizing this root cause completely unlocked project progress—preventing wasteful pursuit of false problems "
        "and demonstrating that combining training data across sources yields a dramatic 3.6x recall boost on unseen cytology slides."
    )

    # Section 2: Chronological Trail of Findings
    add_h1("2. Journey of Discoveries: What Happened & What Was Found")
    
    add_h2("2.1 Phase 1: Baseline Single-Source Model Training")
    doc.add_paragraph(
        "A lightweight YOLOv8n object detection model was initially trained on the 5-class SIPaKMeD dataset "
        "(Superficial-Intermediate, Parabasal, Koilocytotic, Dyskeratotic, and Metaplastic). "
        "Training converged smoothly within 101 epochs (~28 minutes on an RTX 3060 GPU), achieving an initial training-time mAP@50-95 of 0.454. "
        "However, evaluating this model outside its narrow training split immediately exposed key challenges."
    )

    add_h2("2.2 Phase 2: Cross-Dataset Evaluation & Bug Discoveries")
    doc.add_paragraph(
        "When evaluating the model against external cytology sets (APCData and HMCHH-TCT), three critical issues were encountered and solved:"
    )
    b1 = doc.add_paragraph(style='List Bullet')
    b1.add_run("Label Polygon Misparsing: ").bold = True
    b1.add_run(
        "SIPaKMeD labels are stored as polygonal contours (>4 coordinate points) rather than standard bounding boxes. "
        "The initial evaluation code interpreted the first two polygon vertices as width and height, causing intersection-over-union (IoU) matching to fail. "
        "This was corrected across all evaluation modules by dynamically converting polygon coordinate bounds into valid axis-aligned bounding boxes."
    )
    b2 = doc.add_paragraph(style='List Bullet')
    b2.add_run("Class Matrix Sizing: ").bold = True
    b2.add_run(
        "Evaluating a 5-class model on binary datasets (such as HMCHH-TCT) threw index boundary errors because the confusion matrix "
        "was sized solely by target classes rather than the maximum of model and target classes. Sizing was made dynamic."
    )
    b3 = doc.add_paragraph(style='List Bullet')
    b3.add_run("Report Overwrite Protection: ").bold = True
    b3.add_run(
        "Evaluation results initially wrote to a single fixed filename, causing consecutive evaluation runs to overwrite previous datasets. "
        "Distinct, dataset-specific logging and JSON artifacts were established."
    )

    add_h2("2.3 Phase 3: The Label-Sparsity Breakthrough")
    doc.add_paragraph(
        "Initial cross-dataset testing on external APCData slides showed a severe drop in cell recall (from 0.765 down to 0.215), "
        "accompanied by apparent low precision (~0.33 to 0.40) on SIPaKMeD. Visual inspection and independent hematoxylin nucleus counting revealed the truth: "
        "the ground truth labels only cover 3% to 7% of actual squamous epithelial cells present in the images. "
        "Because human curators previously annotated only single isolated cells and ignored large cell sheets, "
        "the model was correctly detecting real cells that lacked ground truth tags—falsely penalizing precision. "
        "Furthermore, because the model was trained on sparse fields, it initially hesitated to fire on dense clusters."
    )

    add_h2("2.4 Multi-Source Training Experiments (C1, C2, C2b)")
    doc.add_paragraph(
        "To break through the single-dataset boundary, a unified 2-class (Normal vs. Abnormal) taxonomy was created, "
        "combining SIPaKMeD and APCData images. The results proved decisive:"
    )

    # Table of experiments
    table = doc.add_table(rows=5, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model Experiment", "Training Data", "Recall (SIPaKMeD)", "Recall (APCData)", "Firing Density Ratio"]
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)
        set_cell_background(cell, "1B365D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 80, 80)
    
    rows_data = [
        ("Phase 1 Baseline", "SIPaKMeD 5-class", "0.765", "0.215", "2.17 / 0.55 (4.0x asymmetry)"),
        ("Exp C1 (Single)", "SIPaKMeD 2-class", "0.807", "0.240", "2.58 / 0.68 (3.8x asymmetry)"),
        ("Exp C2 (Multi-Source)", "SIPaKMeD + APCData", "0.749", "0.860", "1.96 / 2.01 (Balanced)"),
        ("Exp C2b (Replication)", "SIPaKMeD + APCData", "0.765", "0.854", "2.20 / 2.06 (Balanced)")
    ]
    for row_idx, data in enumerate(rows_data, start=1):
        row = table.rows[row_idx]
        for col_idx, val in enumerate(data):
            cell = row.cells[col_idx]
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            set_cell_background(cell, "F9FBFC" if row_idx % 2 == 1 else "FFFFFF")
            set_cell_margins(cell, 60, 60, 80, 80)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    p_takeaway = doc.add_paragraph()
    p_takeaway_run = p_takeaway.add_run(
        "Key Conclusion: Multi-source training boosted cell recall on APCData from 24.0% to 85.4% (a 3.6x increase), "
        "while maintaining identical recall on SIPaKMeD (~76.5%). Cross-domain firing rates balanced completely."
    )
    p_takeaway_run.font.bold = True

    # Section 3: Step-by-Step Execution Completed Today
    add_h1("3. Actions Completed in the Current Session")
    
    add_h2("3.1 Step 1: Generation of the Full 40-Field Dense Annotation Set")
    doc.add_paragraph(
        "To permanently solve the precision measurement barrier, model-assisted pre-annotation was scaled up from the pilot 10 fields "
        "to a full 40-field evaluation cohort (saved in data/dense_eval/sipakmed_dense40). "
        "The detector proposed candidate boxes at a recall-favoring operating threshold, keeping all original ground-truth boxes intact. "
        "A total of 197 original ground-truth boxes were preserved, and 541 high-confidence model candidate boxes were proposed (averaging 18.4 boxes per field, compared to 4.5 in the sparse dataset). "
        "Full assets were compiled including: side-by-side PNG overlay previews, YOLO .txt annotations, Pascal VOC XML files, and a ready-to-import Label Studio task file (labelstudio_tasks.json)."
    )

    add_h2("3.2 Step 2: HMCHH-TCT Over-Firing Root Cause Analysis")
    doc.add_paragraph(
        "In earlier runs, the detector produced 26,315 boxes against 2,054 ground truth labels on HMCHH-TCT (~13:1 over-firing ratio), yielding a low apparent precision of 0.037. "
        "During this session, we confirmed the exact dual mechanism behind this result:\n"
        "1. Protocol Mismatch: The HMCHH-TCT dataset annotates ONLY abnormal/dysplastic cells; normal squamous cells, endocervical cells, and leukocytes are completely unlabeled.\n"
        "2. Detector Generality: Our detector was trained to detect all epithelial cells. When tested on HMCHH-TCT, it successfully detected normal squamous cells in the background, but because the dataset’s ground-truth list omitted them, every valid detected normal cell was penalized as a false positive.\n"
        "Conclusion: The 0.037 precision is an artifact of the dataset's abnormal-only annotation protocol, not a sign of model failure."
    )

    # Section 4: Application Security & Code Audit Report
    add_h1("4. Application Security & Code Audit Report (OWASP 2025)")
    doc.add_paragraph(
        "As an elite Application Security Architect and Code Reviewer, the entire codebase was audited strictly against "
        "the OWASP Top 10: 2025 standard. The repository demonstrates clean design patterns with zero instances of dangerous "
        "command execution (no subprocess shell=True, no eval(), no unsafe pickle deserialization). "
        "The following security vulnerabilities were identified along with immediate, exact remediation code blocks."
    )

    # Vuln 1
    add_h2("🚨 Vulnerability 1: Hardcoded / Staged API Secret in Plaintext .env File (Severity: Medium)")
    doc.add_paragraph("• Location: .env -> Line 1\n• OWASP Category: A02:2025 - Security Misconfiguration / Sensitive Data Exposure")
    doc.add_paragraph(
        "Description: A live Roboflow API key (ROBOFLOW_API_KEY) is currently stored in plaintext on disk in the .env file. "
        "While .env is present in .gitignore, keeping real API tokens unencrypted on local disks risks accidental commits, exposure in automated backups, "
        "or compromise via local read vulnerabilities.\n"
        "Impact: An unauthorized party gaining read access to the directory could consume API quotas, access private cloud workspaces, or tamper with dataset projects.\n"
        "Remediation: Store secrets using system environment variables or secure key vaults, provide a sanitized .env.example, and ensure automated pre-commit scanners reject active keys."
    )
    code_block_1 = (
        "# Remediation Code: download_datasets.py & config.py\n"
        "import os\n"
        "import sys\n"
        "from dotenv import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "def get_secure_roboflow_key() -> str:\n"
        "    api_key = os.environ.get('ROBOFLOW_API_KEY')\n"
        "    if not api_key or api_key.strip() == '' or api_key == 'your_key_here':\n"
        "        raise EnvironmentError(\n"
        "            'ROBOFLOW_API_KEY is not configured in the environment. '\n"
        "            'Please set the environment variable securely or use a protected vault.'\n"
        "        )\n"
        "    return api_key.strip()\n"
    )
    p_code1 = doc.add_paragraph(code_block_1)
    p_code1.paragraph_format.left_indent = Inches(0.4)
    p_code1.runs[0].font.name = 'Consolas'
    p_code1.runs[0].font.size = Pt(9)

    # Vuln 2
    add_h2("🚨 Vulnerability 2: Unhandled Drive I/O Exceptions on External Hardware (Severity: Low)")
    doc.add_paragraph("• Location: config.py -> Path('D:/pap_model/HMCHH_YOLO_prepared')\n• OWASP Category: A10:2025 - Mishandling of Exceptional Conditions")
    doc.add_paragraph(
        "Description: The project configuration defines absolute hardcoded paths to drive D: (e.g. D:/pap_model/HMCHH_YOLO_prepared). "
        "When D: is unmounted, locked by BitLocker, or absent, executing Path.exists() or directory scans raises unhandled OS WinErrors (-2144272384), "
        "halting execution scripts abruptly and revealing internal filesystem structures in tracebacks.\n"
        "Impact: Denial of service for evaluation scripts and verbose stack trace generation.\n"
        "Remediation: Wrap secondary storage checks in defensive path validation helpers with graceful fallbacks."
    )
    code_block_2 = (
        "# Remediation Code: config.py\n"
        "from pathlib import Path\n"
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def safe_resolve_external_path(target_path_str: str) -> Path | None:\n"
        "    try:\n"
        "        path = Path(target_path_str)\n"
        "        if path.exists():\n"
        "            return path\n"
        "        logger.warning(f'External path {target_path_str} does not exist.')\n"
        "        return None\n"
        "    except OSError as e:\n"
        "        logger.warning(f'Secondary drive containing {target_path_str} is inaccessible/locked: {e}')\n"
        "        return None\n"
    )
    p_code2 = doc.add_paragraph(code_block_2)
    p_code2.paragraph_format.left_indent = Inches(0.4)
    p_code2.runs[0].font.name = 'Consolas'
    p_code2.runs[0].font.size = Pt(9)

    # Section 5: Clear Roadmap for Next Work
    add_h1("5. Technical Roadmap & Immediate Next Steps")
    
    doc.add_paragraph(
        "Based on the empirical findings, the remaining work is organized in strict order of priority:"
    )
    
    s1 = doc.add_paragraph(style='List Bullet')
    s1.add_run("1. Human Verification of Dense Eval Set: ").bold = True
    s1.add_run(
        "Launch Label Studio on data/dense_eval/sipakmed_dense40 to conduct a swift human review of the 40 fields. "
        "Because boxes are pre-drawn, annotators only need to delete false alarms and tag missing cells. "
        "Once verified, running the evaluation script against this set will produce the first true, scientifically sound precision & F1 scores."
    )

    s2 = doc.add_paragraph(style='List Bullet')
    s2.add_run("2. Stain Normalization Benchmark (Macenko / Reinhard): ").bold = True
    s2.add_run(
        "Now that the multi-source baseline (C2b) is established and label confounding is isolated, "
        "we can test whether stain normalization produces statistically meaningful recall or mAP gains beyond the baseline ±0.03 run-to-run variation."
    )

    s3 = doc.add_paragraph(style='List Bullet')
    s3.add_run("3. Abnormal-Only Filter for HMCHH-TCT: ").bold = True
    s3.add_run(
        "Evaluate the model against HMCHH-TCT by filtering prediction outputs strictly to abnormal-class categories (Koilocytotic, Dyskeratotic, Abnormal), "
        "aligning the model's output taxonomy directly with the dataset's abnormal-only annotation protocol."
    )

    s4 = doc.add_paragraph(style='List Bullet')
    s4.add_run("4. Automated Pre-Commit Security Hooks: ").bold = True
    s4.add_run(
        "Add a lightweight git hook or scan script that prevents accidental commits of .env files, private model weights, or sensitive patient metadata."
    )

    # Save document
    out_path = r"C:\Users\abusu\Desktop\Pap_Smear_Comprehensive_Audit_and_Progress_Report.docx"
    doc.save(out_path)
    print(f"Report saved successfully to {out_path}")

if __name__ == "__main__":
    create_report()
