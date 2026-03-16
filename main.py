import base64
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import fitz  # PyMuPDF
from supabase import create_client, Client
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .src.detectors.metadata_detector import MetadataDetector
from .src.detectors.pixel_detector import PixelDetector
from .src.detectors.pdf_structure_detector import PDFStructureDetector
from .src.detectors.font_detector import FontDetector
from .src.detectors.text_layer_detector import TextLayerDetector
from .src.detectors.layout_detector import LayoutDetector
from .src.detectors.signature_detector import SignatureDetector
from .src.detectors.embedded_object_detector import EmbeddedObjectDetector
from .src.detectors.confidence_scorer import ConfidenceScorer
from .src.detectors.ai_content_detector import AIContentDetector
from .src.policy_engine import PolicyEngine


app = FastAPI(title="Echt Document Forensics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


policy_engine = PolicyEngine()

# Load .env from backend folder so SUPABASE_URL and SUPABASE_KEY are available
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

print("--- STARTUP CHECK ---")
print(f"SUPABASE_URL found: {bool(os.getenv('SUPABASE_URL'))}")
print(f"SUPABASE_KEY found: {bool(os.getenv('SUPABASE_KEY'))}")

# Supabase client (optional: only if env vars set)
_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None
if _supabase_url and _supabase_key:
    supabase = create_client(_supabase_url, _supabase_key)


def pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    """Convert first page of PDF to PIL Image."""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_page = pdf_doc[0]
    pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()
    return img


def get_file_type(filename: str) -> str:
    """Determine file type from filename."""
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return "pdf"
    if ext in ["jpg", "jpeg", "png", "tiff", "tif", "bmp"]:
        return "image"
    return "unknown"


def run_full_analysis(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Port of `analyze_file` from `src/app.py`, without any Streamlit dependencies.
    """
    file_type = get_file_type(filename)
    file_stream = BytesIO(file_bytes)

    metadata_detector = MetadataDetector()
    pixel_detector = PixelDetector()
    ai_content_detector = AIContentDetector()

    pdf_doc = None
    if file_type == "pdf":
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        structure_detector = PDFStructureDetector()
        font_detector = FontDetector()
        text_layer_detector = TextLayerDetector()
        layout_detector = LayoutDetector()
        signature_detector = SignatureDetector()
        embedded_detector = EmbeddedObjectDetector()
        confidence_scorer = ConfidenceScorer()

    metadata_result = metadata_detector.analyze(BytesIO(file_bytes), file_type, filename)

    if file_type == "pdf":
        display_image = pdf_to_image(file_bytes)
    else:
        file_stream.seek(0)
        display_image = Image.open(file_stream)

    ela_heatmap = pixel_detector.analyze_ela(display_image)
    noise_result = pixel_detector.analyze_noise(display_image)

    ai_result = None
    if file_type == "pdf" and pdf_doc:
        try:
            extracted_text = metadata_result.get("raw_data", {}).get("extracted_text_full", "")
            if not extracted_text:
                text_parts: List[str] = []
                for page_num in range(min(len(pdf_doc), 3)):
                    try:
                        page = pdf_doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception:
                        continue
                extracted_text = "\n\n".join(text_parts)

            if extracted_text and len(extracted_text.strip()) > 50:
                ai_result = ai_content_detector.analyze(extracted_text)
        except Exception:
            ai_result = None

    all_results: Dict[str, Any] = {
        "metadata": metadata_result,
        "pixel": noise_result,
    }

    if ai_result:
        all_results["ai_content"] = ai_result

    if file_type == "pdf" and pdf_doc:
        structure_result = structure_detector.analyze(pdf_doc)
        font_result = font_detector.analyze(pdf_doc)
        text_layer_result = text_layer_detector.analyze(pdf_doc)
        layout_result = layout_detector.analyze(pdf_doc)
        signature_result = signature_detector.analyze(pdf_doc)
        embedded_result = embedded_detector.analyze(pdf_doc)

        all_results.update(
            {
                "structure": structure_result,
                "font": font_result,
                "text_layer": text_layer_result,
                "layout": layout_result,
                "signature": signature_result,
                "embedded": embedded_result,
            }
        )

        confidence_result = confidence_scorer.calculate_confidence(all_results)

        if confidence_result:
            confidence_score = confidence_result["confidence_score"]
            confidence_level = confidence_result["confidence_level"]

            if confidence_level == "Definitive Fraud" and confidence_score >= 90:
                metadata_result["trust_score"] = 15
            elif confidence_level == "High Suspicion" and confidence_score >= 70:
                metadata_result["trust_score"] = 25
            elif confidence_level == "Moderate Suspicion" and confidence_score >= 50:
                metadata_result["trust_score"] = 40
            elif confidence_level == "Low Suspicion" and confidence_score < 30:
                pass
            elif confidence_level == "Low Suspicion" and confidence_score >= 30:
                if metadata_result["trust_score"] > 70:
                    metadata_result["trust_score"] = max(60, metadata_result["trust_score"] - 10)

        if noise_result["flags"]:
            if metadata_result["trust_score"] > 50:
                metadata_result["trust_score"] = 20
            else:
                metadata_result["trust_score"] = min(metadata_result["trust_score"], 20)

        if ai_result and ai_result.get("is_ai_generated") and "ai_content" not in all_results:
            all_results["ai_content"] = ai_result
    else:
        confidence_result = None

    ela_risk_low = not noise_result.get("flags") or len(
        [
            f
            for f in noise_result.get("flags", [])
            if "smoothing" in f.lower() or "ela" in f.lower()
        ]
    ) == 0
    metadata_suspicious = len(metadata_result.get("flags", [])) > 0 or metadata_result.get("risk_score", 0) > 30
    correlation_flag = None
    if ela_risk_low and metadata_suspicious:
        correlation_flag = (
            "⚠️ Correlation: Low image manipulation risk (ELA) but suspicious metadata detected. "
            "Metadata issues may indicate document forgery even without visible image tampering."
        )

    analysis: Dict[str, Any] = {
        "filename": filename,
        "file_type": file_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata_result,
        "noise": noise_result,
        "display_image": display_image,
        "ela_heatmap": ela_heatmap,
        "confidence": confidence_result,
        "ai_content": ai_result,
        "correlation_flag": correlation_flag,
        "all_results": all_results if file_type == "pdf" else {},
    }

    if pdf_doc:
        pdf_doc.close()

    return analysis


def collect_red_flags(analysis: Dict[str, Any]) -> List[str]:
    """
    Aggregate important flags from the analysis into a flat list of messages.
    """
    flags: List[str] = []

    flags.extend(analysis["metadata"].get("flags", []))
    flags.extend(analysis["noise"].get("flags", []))

    ai_content = analysis.get("ai_content")
    if ai_content and ai_content.get("is_ai_generated"):
        confidence = ai_content.get("confidence", 0)
        flags.append(f"AI-generated content detected (confidence {confidence:.1f}%).")
        for indicator in ai_content.get("indicators", []):
            flags.append(indicator)

    if analysis.get("correlation_flag"):
        flags.append(analysis["correlation_flag"])

    if analysis["file_type"] == "pdf" and analysis.get("all_results"):
        all_results = analysis["all_results"]
        for key in ["structure", "font", "text_layer", "layout", "signature", "embedded"]:
            if key in all_results:
                flags.extend(all_results[key].get("flags", []))

    metadata_raw = analysis["metadata"].get("raw_data", {})
    assessment = metadata_raw.get("assessment")
    if assessment:
        flags.append(f"Assessment: {assessment}")

    seen = set()
    deduped: List[str] = []
    for f in flags:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return deduped


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...), doc_type_key: str = Form(...)) -> Dict[str, Any]:
    """
    Analyze an uploaded document and return key forensic metrics and policy verdict.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        file_bytes = await file.read()
        analysis = run_full_analysis(file.filename, file_bytes)
        policy_result = policy_engine.evaluate(analysis, doc_type_key)

        forgery_score = analysis["metadata"].get("risk_score", 0)
        trust_score = analysis["metadata"].get("trust_score", 0)
        red_flags = collect_red_flags(analysis)

        preview_image_base64 = None
        preview_image_media_type = None
        display_image = analysis.get("display_image")
        if display_image is not None:
            buf = BytesIO()
            display_image.save(buf, format="PNG")
            preview_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            preview_image_media_type = "image/png"

        ela_image_base64 = None
        ela_heatmap = analysis.get("ela_heatmap")
        if ela_heatmap is not None:
            buf_ela = BytesIO()
            if ela_heatmap.mode != "RGB":
                ela_heatmap = ela_heatmap.convert("RGB")
            ela_heatmap.save(buf_ela, format="PNG")
            ela_image_base64 = base64.b64encode(buf_ela.getvalue()).decode("utf-8")

        ai_content = analysis.get("ai_content") or {}
        ai_confidence = ai_content.get("confidence")
        if ai_confidence is not None:
            ai_confidence = round(float(ai_confidence), 1)
        ai_indicators = ai_content.get("indicators") or []

        extracted_text = ""
        metadata_raw = (analysis.get("metadata") or {}).get("raw_data") or {}
        if metadata_raw.get("extracted_text_full"):
            raw_text = metadata_raw["extracted_text_full"]
            extracted_text = raw_text[:10000] if len(raw_text) > 10000 else raw_text

        noise_findings = (analysis.get("noise") or {}).get("findings") or ""

        detector_summary: Dict[str, Any] = {}
        all_results = analysis.get("all_results") or {}
        for key in ["structure", "font", "text_layer", "layout", "signature", "embedded"]:
            if key not in all_results:
                continue
            r = all_results[key]
            detector_summary[key] = {
                "flags": r.get("flags") or [],
                "risk_score": r.get("risk_score"),
            }

        # File DNA: forensic properties for the File DNA tab
        file_dna: List[Dict[str, str]] = []
        try:
            metadata_result = analysis.get("metadata") or {}
            metadata_raw = metadata_result.get("raw_data") or {}
            pdf_meta = metadata_raw.get("pdf_metadata") or {}
            noise_result = analysis.get("noise") or {}
            noise_var = noise_result.get("variance")
            inst_indicators = metadata_raw.get("institutional_indicators")
            if isinstance(inst_indicators, list):
                inst_str = ", ".join(str(x) for x in inst_indicators) if inst_indicators else "—"
            else:
                inst_str = str(inst_indicators) if inst_indicators else "—"
            creation_date = pdf_meta.get("creationDate") or ""
            mod_date = pdf_meta.get("modDate") or ""
            file_dna = [
                {"property": "Author", "value": (metadata_raw.get("author") or "").strip() or "—"},
                {"property": "Creator", "value": (metadata_raw.get("creator") or "").strip() or "—"},
                {"property": "Producer", "value": (metadata_raw.get("producer") or "").strip() or "—"},
                {"property": "Creation Date", "value": (creation_date.strip() if isinstance(creation_date, str) else str(creation_date)) or "—"},
                {"property": "Modification Date", "value": (mod_date.strip() if isinstance(mod_date, str) else str(mod_date)) or "—"},
                {"property": "Institutional Indicators", "value": inst_str or "—"},
                {"property": "Noise Variance", "value": f"{noise_var:.2f}" if noise_var is not None else "—"},
            ]
        except Exception:
            file_dna = [
                {"property": "Note", "value": "File DNA could not be built for this document."},
            ]

        out: Dict[str, Any] = {
            "filename": analysis["filename"],
            "doc_type_key": doc_type_key,
            "forgery_score": forgery_score,
            "trust_score": trust_score,
            "red_flags": red_flags,
            "policy_result": policy_result,
            "timestamp": analysis["timestamp"],
            "file_dna": file_dna,
            "detector_summary": detector_summary,
            "extracted_text": extracted_text,
            "noise_findings": noise_findings,
        }
        if preview_image_base64 is not None:
            out["preview_image_base64"] = preview_image_base64
            out["preview_image_media_type"] = preview_image_media_type
        if ela_image_base64 is not None:
            out["ela_image_base64"] = ela_image_base64
        if ai_confidence is not None:
            out["ai_confidence"] = ai_confidence
        if ai_indicators:
            out["ai_indicators"] = ai_indicators

        # Persist scan result to Supabase (non-blocking; do not fail the request)
        print("--- PREPARING DATABASE INSERT ---")
        if supabase is None:
            print("WARNING: Supabase client is None! Skipping insert.")
        else:
            print("Supabase client exists. Attempting insert...")
        if supabase is not None:
            try:
                verdict = (policy_result or {}).get("verdict", "")
                supabase.table("scans").insert(
                    {
                        "filename": file.filename or out.get("filename", ""),
                        "doc_type": doc_type_key,
                        "verdict": verdict,
                        "forgery_score": forgery_score,
                        "trust_score": trust_score,
                        "red_flags": red_flags,
                    }
                ).execute()
            except Exception as e:
                print(f"SUPABASE ERROR: {e}")
                logging.exception("Supabase insert failed: %s", e)

        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

