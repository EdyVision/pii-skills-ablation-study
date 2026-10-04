"""Pinned scoring contract for archived report results, with explicit gold provenance.
No downloads, inference, or mutation of raw run files.
"""
import json
from pathlib import Path
from collections import Counter
from ablation_harness.scoring.diagnostics import validate_records
GROUND_TRUTH_TO_PIICODEX = json.loads((Path(__file__).resolve().parents[3] / "config/pii_label_to_piicodex.json").read_text())
PREDICTION_TO_PIICODEX = {
    # Identity (model outputs canonical PII Codex type name)
    "US_BANK_ACCOUNT_NUMBER": "US_BANK_ACCOUNT_NUMBER",
    "US_DRIVERS_LICENSE_NUMBER": "US_DRIVERS_LICENSE_NUMBER",
    "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION": "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION",
    "NRP": "NRP",  # Nationality, religion, political (PII Codex PIIType)
    # Driver / license
    "DRIVER_LICENSE_NUMBER": "US_DRIVERS_LICENSE_NUMBER",
    "US_DRIVER_LICENSE": "US_DRIVERS_LICENSE_NUMBER",
    "DRIVER'S_LICENSE": "US_DRIVERS_LICENSE_NUMBER",
    # SSN
    "SOCIAL_SECURITY_NUMBER": "US_SOCIAL_SECURITY_NUMBER",
    # Postal / zip
    "POSTAL_CODE": "ZIPCODE",
    "US_POSTAL_CODE": "ZIPCODE",
    # Location (COUNTRY_OF_ORIGIN -> LOCATION per paper Appendix A.2)
    "COUNTRY_OF_ORIGIN": "LOCATION",
    # Person / account
    "ACCOUNT_USERNAME": "PERSON",
    "SIBLING_NAME": "PERSON",
    "INDIVIDUAL": "PERSON",
    # Medical / health IDs (PII Codex: HEALTH_INSURANCE_ID)
    "MRN": "HEALTH_INSURANCE_ID",
    "MRN_NUMBER": "HEALTH_INSURANCE_ID",
    "PATIENT_ID": "HEALTH_INSURANCE_ID",
    "PATIENTID": "HEALTH_INSURANCE_ID",
    # ID document
    "ID_CARD_NUMBER": "US_PASSPORT_NUMBER",
    # Biometric (PII Codex: FINGERPRINT)
    "BIOM_IDENTIFIER": "FINGERPRINT",
    # Time
    "TIME_OF_DAY": "DATE_TIME",
    "SESSION_TIME": "DATE_TIME",
}

combined_label_map = {**GROUND_TRUTH_TO_PIICODEX, **PREDICTION_TO_PIICODEX}
ALIGNMENT_OVERRIDES = {
    "DATE": "DATE_TIME", "DATEOFBIRTH": "DATE_TIME", "DATETIME": "DATE_TIME", "DATE_AND_TIME": "DATE_TIME", "TIMESTAMP": "DATE_TIME",
    "ADDRESS": "LOCATION", "STREET": "LOCATION", "STREET_ADDRESS": "LOCATION", "STREET_NAME": "LOCATION",
    "BUILDING": "LOCATION", "BUILDING_NUMBER": "LOCATION", "SECONDARY_ADDRESS": "LOCATION", "SECADDRESS": "LOCATION",
    "UK_NHS": "HEALTH_INSURANCE_ID", "MEDICALRECORDNUMBER": "HEALTH_INSURANCE_ID",
    "GENDER": "GENDER", "IPV4_ADDRESS": "IP_ADDRESS", "IPV6_ADDRESS": "IP_ADDRESS",
    "ZIP_CODE": "ZIPCODE", "ZIP": "ZIPCODE", "POSTALCODE": "ZIPCODE",
    "SOCIALSECURITYNUMBER": "US_SOCIAL_SECURITY_NUMBER", "SOCIAL_NUMBER": "US_SOCIAL_SECURITY_NUMBER",
    "PASSPORTNUMBER": "US_PASSPORT_NUMBER", "TELEPHONE_NUMBER": "PHONE_NUMBER",
}
CANONICAL_ALIAS = {**combined_label_map, **ALIGNMENT_OVERRIDES}
import re as _re

def _key(t): return t.upper().replace("-", "_").replace(" ", "_") if t else ""

def _fallback(k):
    if _re.search(r"IPV?4|IPV?6|IP_?ADDR", k): return "IP_ADDRESS"
    if "ZIP" in k or "POSTAL" in k: return "ZIPCODE"
    if "SSN" in k or "SOCIAL_SEC" in k or "SOCIALSEC" in k: return "US_SOCIAL_SECURITY_NUMBER"
    if "PASSPORT" in k: return "US_PASSPORT_NUMBER"
    if "DRIVER" in k and "LICEN" in k: return "US_DRIVERS_LICENSE_NUMBER"
    if k.startswith("DATE") or k.endswith("DATE") or "BIRTH" in k or k in ("TIME",) or "TIMESTAMP" in k: return "DATE_TIME"
    if "EMAIL" in k: return "EMAIL_ADDRESS"
    if "PHONE" in k or "TELEPHONE" in k or k in ("TEL", "MOBILE", "FAX"): return "PHONE_NUMBER"
    if "ADDRESS" in k or k in ("CITY", "STATE", "COUNTRY", "COUNTY", "STREET"): return "LOCATION"

    return None

def norm(t):
    k = _key(t)
    if not k: return "UNKNOWN"
    if k in CANONICAL_ALIAS:
        v = CANONICAL_ALIAS[k]; return v if v is not None else "UNMAPPED:" + k

    return _fallback(k) or k

def norm_pre(t):  # base map used symmetrically before expanded alignment
    k = _key(t); v = combined_label_map.get(k, k)

    return v if v is not None else "UNKNOWN"

def _sint(d, k):
    v = d.get(k, 0)

    return v if isinstance(v, int) else (int(v) if str(v).lstrip("-").isdigit() else 0)

def _iou(a, b):
    s1, e1, s2, e2 = _sint(a,"start"), _sint(a,"end"), _sint(b,"start"), _sint(b,"end")
    inter = max(0, min(e1, e2) - max(s1, s2)); u = (e1-s1) + (e2-s2) - inter

    return inter / u if u > 0 else 0.0

def f1(preds, gt, nf, span):
    vp = [p for p in preds if isinstance(p, dict) and "type" in p]
    if not gt: return 1.0 if not vp else 0.0
    if not vp: return 0.0
    used, tp = set(), 0

    for p in vp:
        pt = nf(p.get("type", ""))
        for i, t in enumerate(gt):
            if i in used or pt != nf(t.get("type", "")): continue
            if span and (_sint(t,"start") or _sint(t,"end")) and _iou(p, t) < 0.5: continue
            used.add(i); tp += 1; break

    pr, rc = tp/len(vp), tp/len(gt)

    return 2*pr*rc/(pr+rc) if (pr+rc) else 0.0

def mean_f1(recs, cond, nf, span):
    v = [f1(r["predictions"] or [], r["ground_truth"] or [], nf, span) for r in recs if r["condition"] == cond]

    return sum(v)/len(v) if v else float("nan")

def load_saved_run(root, run_type):
    path = Path(root) / "results/past_runs" / run_type / "experiment_results.json"
    records = json.loads(path.read_text())
    validate_records(records)
    if run_type == "detector":
        # The archived detector file omitted fallback gold on 868 records.
        main = json.loads((Path(root) / "results/past_runs/main/experiment_results.json").read_text())
        by_id = {}
        for row in main:
            sid = row["sample_id"]
            if sid in by_id and row["ground_truth"] != by_id[sid]["ground_truth"]:
                raise ValueError("Inconsistent primary ground truth")
            by_id[sid] = row
        if {r["sample_id"] for r in records} != set(by_id):
            raise ValueError("Detector and primary sample sets differ")
        records = [{**r, "ground_truth": by_id[r["sample_id"]]["ground_truth"],
                    "gold_provenance": "matched primary saved gold by sample_id"} for r in records]
    if any(not row.get("ground_truth") for row in records):
        raise ValueError("Missing gold is not a negative example in this benchmark")
    return records


def type_metrics(predictions, gold, normalize=norm):
    preds = [p for p in predictions if isinstance(p, dict) and "type" in p]
    pc = Counter(normalize(p.get("type", "")) for p in preds)
    gc = Counter(normalize(g.get("type", "")) for g in gold)
    tp = sum((pc & gc).values())
    if not gold:
        return {"precision": float(not preds), "recall": 1., "f1": float(not preds)}
    return {"precision": tp / len(preds) if preds else 0., "recall": tp / len(gold),
            "f1": 2 * tp / (len(preds) + len(gold))}
