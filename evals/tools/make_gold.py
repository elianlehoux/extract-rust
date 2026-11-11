#!/usr/bin/env python3
"""Build evals/gold.jsonl from hand-written annotations below.

Entities are listed in order of appearance; offsets are computed as Unicode
code-point indices (the same convention as LangExtract and extract-rust) by
searching forward from the end of the previous entity, and the script fails
if any span is missing. Run: python3 evals/tools/make_gold.py
"""
import json
import pathlib

DOCS = [
    # ---------------- general: person / organization / location ----------------
    ("gen-01", "general", "en",
     "Satya Nadella announced that Microsoft will open a new data center in Querétaro next spring.",
     [("person", "Satya Nadella"), ("organization", "Microsoft"), ("location", "Querétaro")]),
    ("gen-02", "general", "en",
     "After leaving NASA, Mae Jemison founded The Jemison Group in Houston.",
     [("organization", "NASA"), ("person", "Mae Jemison"), ("organization", "The Jemison Group"),
      ("location", "Houston")]),
    ("gen-03", "general", "en",
     "Lionel Messi signed with Inter Miami and moved from Paris to Florida in 2023.",
     [("person", "Lionel Messi"), ("organization", "Inter Miami"), ("location", "Paris"),
      ("location", "Florida")]),
    ("gen-04", "general", "es",
     "La ingeniera Sofía Martínez se sumó a Mercado Libre en Buenos Aires para liderar el equipo de pagos.",
     [("person", "Sofía Martínez"), ("organization", "Mercado Libre"), ("location", "Buenos Aires")]),
    ("gen-05", "general", "es",
     "El rector de la Universidad Nacional de Córdoba, Jhon Boretto, recibió a una delegación de la UNESCO en Córdoba.",
     [("organization", "Universidad Nacional de Córdoba"), ("person", "Jhon Boretto"),
      ("organization", "UNESCO"), ("location", "Córdoba")]),
    ("gen-06", "general", "es",
     "Marcos Galperin y Hernán Kazah fundaron la empresa en 1999 antes de expandirse a México y Brasil.",
     [("person", "Marcos Galperin"), ("person", "Hernán Kazah"), ("location", "México"),
      ("location", "Brasil")]),
    # ---------------- medical: medication / dosage / frequency ----------------
    ("med-01", "medical", "en",
     "The patient was started on metformin 500 mg twice daily and advised to avoid alcohol.",
     [("medication", "metformin"), ("dosage", "500 mg"), ("frequency", "twice daily")]),
    ("med-02", "medical", "en",
     "Continue lisinopril 10 mg once daily; add atorvastatin 20 mg at bedtime.",
     [("medication", "lisinopril"), ("dosage", "10 mg"), ("frequency", "once daily"),
      ("medication", "atorvastatin"), ("dosage", "20 mg"), ("frequency", "at bedtime")]),
    ("med-03", "medical", "en",
     "For the headache she took ibuprofen 400 mg every 8 hours for two days.",
     [("medication", "ibuprofen"), ("dosage", "400 mg"), ("frequency", "every 8 hours")]),
    ("med-04", "medical", "es",
     "Se indicó amoxicilina 875 mg cada 12 horas durante siete días.",
     [("medication", "amoxicilina"), ("dosage", "875 mg"), ("frequency", "cada 12 horas")]),
    ("med-05", "medical", "es",
     "El paciente toma enalapril 5 mg una vez por día y paracetamol 1 g cada 8 horas si hay dolor.",
     [("medication", "enalapril"), ("dosage", "5 mg"), ("frequency", "una vez por día"),
      ("medication", "paracetamol"), ("dosage", "1 g"), ("frequency", "cada 8 horas")]),
    # ---------------- metar: station / wind / visibility / clouds / temp / pressure ----------------
    ("metar-01", "metar", "en",
     "METAR SAAR 181200Z 27015G25KT 9999 SCT030 18/09 Q1013",
     [("station", "SAAR"), ("wind", "27015G25KT"), ("visibility", "9999"), ("clouds", "SCT030"),
      ("temperature", "18/09"), ("pressure", "Q1013")]),
    ("metar-02", "metar", "en",
     "METAR KJFK 181551Z 31012KT 10SM FEW250 14/M02 A3012",
     [("station", "KJFK"), ("wind", "31012KT"), ("visibility", "10SM"), ("clouds", "FEW250"),
      ("temperature", "14/M02"), ("pressure", "A3012")]),
    ("metar-03", "metar", "en",
     "METAR SACO 180900Z 36008KT 4000 BR OVC006 12/11 Q1018",
     [("station", "SACO"), ("wind", "36008KT"), ("visibility", "4000"), ("clouds", "OVC006"),
      ("temperature", "12/11"), ("pressure", "Q1018")]),
    ("metar-04", "metar", "en",
     "METAR EGLL 181020Z 22018KT 8000 BKN012 11/08 Q0998",
     [("station", "EGLL"), ("wind", "22018KT"), ("visibility", "8000"), ("clouds", "BKN012"),
      ("temperature", "11/08"), ("pressure", "Q0998")]),
    # ---------------- flight_report: flight_number / aircraft / airport ----------------
    ("fr-01", "flight_report", "en",
     "Flight AR1550 operated by an Embraer E190 diverted from Ezeiza to Rosario due to fog.",
     [("flight_number", "AR1550"), ("aircraft", "Embraer E190"), ("airport", "Ezeiza"),
      ("airport", "Rosario")]),
    ("fr-02", "flight_report", "en",
     "The Boeing 737-800 operating LA2431 returned to Santiago after a bird strike on departure.",
     [("aircraft", "Boeing 737-800"), ("flight_number", "LA2431"), ("airport", "Santiago")]),
    ("fr-03", "flight_report", "en",
     "UA857 from San Francisco landed in Shanghai Pudong on a Boeing 777-300ER.",
     [("flight_number", "UA857"), ("airport", "San Francisco"), ("airport", "Shanghai Pudong"),
      ("aircraft", "Boeing 777-300ER")]),
    ("fr-04", "flight_report", "es",
     "El vuelo FO5120, operado con un Boeing 737 MAX 8, aterrizó en Aeroparque tras despegar de Mendoza.",
     [("flight_number", "FO5120"), ("aircraft", "Boeing 737 MAX 8"), ("airport", "Aeroparque"),
      ("airport", "Mendoza")]),
    ("fr-05", "flight_report", "es",
     "Un Cessna 172 con matrícula LV-FQT despegó de San Fernando y aterrizó en Paraná sin novedades.",
     [("aircraft", "Cessna 172"), ("aircraft", "LV-FQT"), ("airport", "San Fernando"),
      ("airport", "Paraná")]),
]


def main() -> None:
    out = pathlib.Path(__file__).resolve().parents[1] / "gold.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for doc_id, domain, lang, text, ents in DOCS:
            cursor = 0
            extractions = []
            for cls, span in ents:
                start = text.find(span, cursor)
                if start < 0:
                    raise SystemExit(f"{doc_id}: span {span!r} not found after {cursor}")
                end = start + len(span)
                extractions.append({
                    "extraction_class": cls,
                    "extraction_text": span,
                    "char_interval": {"start_pos": start, "end_pos": end},
                })
                cursor = end
            f.write(json.dumps({"id": doc_id, "domain": domain, "lang": lang, "text": text,
                                "extractions": extractions}, ensure_ascii=False) + "\n")
    n = sum(len(d[4]) for d in DOCS)
    print(f"wrote {out} ({len(DOCS)} docs, {n} entities)")


if __name__ == "__main__":
    main()
