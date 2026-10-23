#!/usr/bin/env python3
"""Inventory digit-bearing scientific occurrences in the active paper.

This is a provenance inventory, not a numerical verifier. Repeated occurrences
remain separate; only explicit numeric macros share a quantity identifier.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if ROOT.name == 'tools':
    ROOT = ROOT.parent
NUMBER = r"(?:\d{1,3}(?:\{,\}\d{3}|,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
EXPONENT = r"(?:\{\s*[+−-]?\d+\s*\}|[+−-]?\d+)"
NUMERIC = re.compile(
    rf"(?:(?<![A-Za-z0-9])|(?<=\\pm)|(?<=\\ge)|(?<=\\le)|(?<=\\approx))(?:"
    rf"[−-]?{NUMBER}\s*\\(?:times|cdot)\s*10\s*\^\s*{EXPONENT}"
    rf"|10\s*\^\s*{EXPONENT}"
    rf"|[−-]?{NUMBER}(?:[eE][+−-]?\d+)"
    rf"|[−-]?{NUMBER}\s*/\s*{NUMBER}"
    rf"|[−-]?{NUMBER})(?:\s*\\%)?(?!\d)"
)
COMMAND = re.compile(r"\\[A-Za-z@]+\*?")
LABEL = re.compile(r"\\label\s*\{([^{}]+)\}")
SECTION = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{")
FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}")
COUNT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
               "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
               "fourteen": 14, "fifteen": 15, "sixteen": 16,
               "seventeen": 17, "eighteen": 18, "nineteen": 19,
               "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
               "sixty": 60, "seventy": 70, "eighty": 80,
               "ninety": 90, "hundred": 100, "fourth": 4}
COUNT = re.compile(r"\b(" + "|".join(COUNT_WORDS) + r")\b", re.I)
COUNT_NOUN = re.compile(r"(?:roots?|crossings?|seeds?|runs?|points?|features?|directions?|settings?|cases?|"
                        r"orders?|standard[ -]deviations?|decades?|blocks?|slices?|rows?|"
                        r"centres?|subspaces?|halves|columns?|families|equations?|architectures?|"
                        r"conditions?|modes?|steps?|parameters?|probabilities|intervals?|sigma)\b", re.I)


@dataclass
class Token:
    position: int
    stop: int
    text: str
    form: str = "DIGIT"

    def start(self):
        return self.position

    def end(self):
        return self.stop

    def __getitem__(self, key):
        assert key == 0
        return self.text


def number_tokens(text, begin):
    result = []
    for match in NUMERIC.finditer(text, begin):
        # An unprotected comma inside a tuple/array separates entries. TeX's
        # {,} thousands separators remain a single scientific number.
        left = text[max(begin, match.start() - 160):match.start()]
        array_open = max(left.rfind("["), left.rfind("("))
        array_close = max(left.rfind("]"), left.rfind(")"))
        if "," in match[0] and "{,}" not in match[0] and array_open > array_close:
            for item in re.finditer(r"[−-]?\d+(?:\.\d+)?", match[0]):
                result.append(Token(match.start() + item.start(), match.start() + item.end(), item[0]))
        else:
            result.append(Token(match.start(), match.end(), match[0]))
    for match in COUNT.finditer(text, begin):
        # Capture relevant verbal counts, not every rhetorical use of 'one'.
        tail = text[match.end():match.end() + 100]
        noun = COUNT_NOUN.search(tail)
        immediate = bool(re.match(r"(?:[- ]standard[- ]deviation|[- ]subspace|[- ]order)", tail, re.I))
        between = tail[:noun.start()] if noun else ""
        if immediate or (noun and len(between.split()) <= 6 and not re.search(r"[.;!?]|\n\s*\n", between)):
            result.append(Token(match.start(), match.end(), match[0], "RELEVANT_WORD_COUNT"))
    return sorted(result, key=lambda item: item.start())


def group(text, opening):
    """Return a balanced brace group's contents and its exclusive end."""
    assert text[opening] == "{"
    depth = 1
    for i in range(opening + 1, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[opening + 1:i], i + 1
    raise ValueError(f"Unclosed brace group at offset {opening}")


def uncomment(text):
    # Replace comments with spaces to retain physical line coordinates.
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m[0]), text)


def plain(text):
    text = re.sub(r"\\(?:ref|eqref|pageref|label|cite\w*)\*?(?:\[[^]]*\])?\{[^{}]*\}", " ", text)
    text = text.replace("{,}", ",").replace(r"\%", "%")
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathcal|operatorname|emph|textbf|textit)\s*\{", "{", text)
    text = re.sub(r"\\(?:times|cdot)", "×", text)
    text = re.sub(r"\\[A-Za-z@]+\*?", " ", text)
    text = re.sub(r"[{}$]", "", text).replace(r"\,", " ")
    return re.sub(r"\s+", " ", text).strip()


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Span:
    start: int
    end: int
    source: str
    line: int
    source_text: str
    macro: str = ""
    definition: str = ""


class Paper:
    def __init__(self, root):
        self.root = root
        self.parts = []
        self.spans = []
        self.macros = {}
        self.files = {}
        self.includes = []
        self.offset = 0
        self.stack = []

    def append(self, text, source, source_offset, source_text, macro="", definition=""):
        if not text:
            return
        # A span per physical source line keeps later lookups simple and exact.
        original_line = source_text.count("\n", 0, source_offset) + 1
        for part in text.splitlines(keepends=True):
            self.parts.append(part)
            self.spans.append(Span(self.offset, self.offset + len(part), source,
                                   original_line, part, macro, definition))
            self.offset += len(part)
            if not macro:
                original_line += part.count("\n")

    def normal(self, text, source, source_offset, source_text):
        cursor = 0
        for match in re.finditer(r"\\([A-Za-z@]+)\b", text):
            name = match[1]
            if name not in self.macros or not re.search(r"\d", self.macros[name][0]):
                continue
            self.append(text[cursor:match.start()], source, source_offset + cursor, source_text)
            value, definition = self.macros[name]
            self.append(value, source, source_offset + match.start(), source_text,
                        name, definition)
            cursor = match.end()
        self.append(text[cursor:], source, source_offset + cursor, source_text)

    def read(self, path):
        path = path.resolve()
        path.relative_to(self.root.resolve())
        if path in self.stack:
            raise ValueError(f"Recursive include: {path}")
        self.stack.append(path)
        source = path.relative_to(self.root).as_posix()
        raw = path.read_text(encoding="utf-8")
        text = uncomment(raw)
        self.files[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        cursor = 0
        pattern = re.compile(r"\\(?:input|include)\s*\{|\\(?:newcommand|renewcommand|providecommand)\*?\s*\{")
        for match in pattern.finditer(text):
            if match.start() < cursor:
                continue
            self.normal(text[cursor:match.start()], source, cursor, raw)
            value, end = group(text, match.end() - 1)
            if match[0].startswith((r"\input", r"\include")):
                included = self.root / value
                if not included.suffix:
                    included = included.with_suffix(".tex")
                self.includes.append({"from": source, "line": raw.count("\n", 0, match.start()) + 1,
                                      "included": included.relative_to(self.root).as_posix()})
                self.read(included)
            else:
                tail = end
                while tail < len(text) and text[tail].isspace():
                    tail += 1
                argument_count = 0
                if tail < len(text) and text[tail] == "[":
                    close = text.index("]", tail)
                    argument_count = int(text[tail + 1:close])
                    tail = close + 1
                    while tail < len(text) and text[tail].isspace():
                        tail += 1
                if tail < len(text) and text[tail] == "{":
                    body, end = group(text, tail)
                    if argument_count == 0:
                        definition = f"paper/{source}:{raw.count(chr(10), 0, match.start()) + 1}"
                        self.macros[value.lstrip("\\")] = (body, definition)
            cursor = end
        self.normal(text[cursor:], source, cursor, raw)
        self.stack.pop()

    def finish(self):
        self.text = "".join(self.parts)
        self.starts = [span.start for span in self.spans]

    def origin(self, position):
        return self.spans[bisect.bisect_right(self.starts, position) - 1]


TABLES = [
    ("tab:rejected", "training/analysis/solve_gate_table.py", "provenance/training_configs.json; training/results/*/*_map.json", "All attempted configurations; admission metadata, not independent rejected-field recomputation."),
    ("tab:ceiling|tab:absolute", "analysis/round2/review_production.py; analysis/round2/report_saved_summaries.py; training/analysis/ceiling_table.py", "analysis/round2/outputs/production/*_round2_fields.npz; analysis/round2/outputs/production_review/regional_scale_metrics.csv", "Current family means/ranges and separate linear-control records."),
    ("tab:temporal", "training/analysis/temporal_per_seed.py", "analysis/n5/input/burgers_mlp_seed*/burgers_mlp_seed*_fields.npz; analysis/round2/baseline/*_fullgrid_sigma_raw.npz", "Eight seeds; original global masks intersected with t < 1/pi or t >= 1/pi."),
    ("tab:offmanifold", "training/d4_manifold.py; training/analysis/offmanifold_summary.py; analysis/n5/n5_heldout.py; analysis/n5/summarise_n5.py", "analysis/n5/input/*/*_params.pkl; analysis/n5/input/*/*_map.json; analysis/n5/input/*/*_fields.npz; analysis/n5/outputs", "Same-grid and held-out last-layer fractions; checkpoint refits are needed for independent reproduction."),
    ("tab:spectrum|tab:loading", "training/d_loading_perpoint.py; training/d_loading_cdf.py; training/analysis/loading_table.py; training/analysis/loading_table_tex.py; analysis/recompute_saved_summaries.py", "paper/generated/loading_table_values.tex; paper/generated/rms_bounds_macros.tex; original loading/spectral JSON", "Returned Gram spectrum and bound arithmetic have different numerical verification levels."),
    ("tab:n1-scales", "analysis/n1/n1_sensitivity.py; analysis/n1/summarize_n1.py", "analysis/n1/outputs_cpu/burgers_mlp_seed*_n1.csv; analysis/n1/outputs_cpu/*_n1.json", "Per-seed Burgers–MLP core study, not all-17 family aggregation."),
    ("tab:r2-limits", "analysis/round2/review_production.py; analysis/round2/generate_tables.py", "analysis/round2/outputs/production/*_round2_spectral.npz; analysis/round2/outputs/production_review/spectral_limits.csv", "Conditional exact-kernel limits versus returned-spectrum/QR-complement numerical quantities."),
    ("tab:r2-evidence-summary", "analysis/round2/review_evidence.py; analysis/round2/review_production.py", "analysis/round2/outputs/production/*_round2_spectral.npz; analysis/round2/outputs/production_review/evidence_best.csv; analysis/round2/outputs/production_review/evidence_perturbations.csv", "Finite-grid optima, boundary flags and bounded perturbation checks."),
    ("tab:r2-centres", "analysis/theta_centred/scripts/summarize_theta.py; analysis/theta_centred/scripts/make_paper_tables.py", "analysis/theta_centred/outputs/production/*_theta_fields.npz; analysis/theta_centred/outputs/review/full_grid_errors.csv", "Paired family error means and worse-than-trained counts for each predictive centre."),
    ("tab:r2-relative-shift", "analysis/round2/report_saved_summaries.py; analysis/round2/review_production.py", "analysis/round2/outputs/production_review/n3_centre_comparison.csv; analysis/round2/outputs/reporting_closure/regional_relative_shift.csv", "Zero-mean-prior relative shifts; ensure underlying field review was rerun."),
    ("tab:r2-oracle", "analysis/round2/review_production.py; analysis/round2/generate_tables.py", "analysis/round2/outputs/production/*_round2_fields.npz; analysis/round2/outputs/production_review/nearest_sampled_agreement.csv", "Saved-setting oracle, separately selected for each centre/subset."),
    ("tab:r3-roots", "analysis/round3_local/review_round3.py; analysis/theta_centred/scripts/summarize_theta.py; analysis/theta_centred/scripts/make_paper_tables.py", "analysis/round3_local/outputs; analysis/theta_centred/inputs/refined_roots.csv; analysis/theta_centred/outputs/review/root_comparisons.csv", "Refined modal roots, mathematical coverage envelopes and direct trained-root fields."),
    ("tab:r4-oracles", "analysis/round4_local/review_round4.py", "analysis/round2/outputs/production/*_round2_fields.npz; analysis/round4_local/outputs", "Bulk/sharp calibration, distribution summaries and sampled agreement band."),
    ("tab:r2-evidence", "analysis/round2/review_evidence.py; analysis/round2/generate_tables.py", "analysis/round2/outputs/production_review/evidence_best.csv; analysis/round2/outputs/production_review/evidence_perturbations.csv", "Per-run original and extended affine-evidence selections."),
    ("tab:r2-changed-bounds", "analysis/round2/generate_tables.py", "analysis/round2/outputs/production/*_round2_spectral.npz; analysis/round2/outputs/production_review", "Changed-scale spectral bounds at recorded evidence choices."),
    ("tab:r2-scale-loadings", "analysis/round2/report_saved_summaries.py", "analysis/round2/outputs/reporting_closure/changed_scale_loadings.csv; analysis/round2/outputs/production/*_round2_spectral.npz", "Scale-dependent loadings from returned spectral modes."),
    ("tab:oa-relative-shift", "analysis/round2/report_saved_summaries.py", "analysis/round2/outputs/reporting_closure/regional_relative_shift.csv", "Baseline, MLE and both evidence grids; relative to each region's original error."),
    ("tab:lean", "lean/PaperReliability.lean; lean/Audit.lean", "lean/PaperReliability/*.lean; lean/THEOREM_MAP.md; lean/lean-toolchain; lean/lake-manifest.json", "Statement/declaration map. Declaration existence is not proof rebuilding; empirical counts are not Lean-certified."),
]
FIGURES = [
    ("fig:burgers-slice", "analysis/round2/plot_shortened_slice.py; tools/export_burgers_slice.py", "analysis/burgers_slice.csv; analysis/n5/input/burgers_mlp_seed2/burgers_mlp_seed2_fields.npz; analysis/round2/baseline/burgers_mlp_seed2_fullgrid_sigma_raw.npz", "400 spatial values, time index 100; raw covariance field, not checkpoint historical sigma."),
    ("fig:n1-scales", "analysis/round2/plot_shortened_records.py", "analysis/round2/outputs/production_review/uncertainty_contrast_curves.csv; analysis/round2/outputs/production_review/regional_scale_metrics.csv", "Replot of reviewed modal and field metrics; PDF byte identity is not required."),
    ("fig:r2-evidence-energy", "analysis/round2/plot_shortened_records.py", "analysis/round2/outputs/production_review/evidence_slope_groups.csv; analysis/round2/outputs/production/*_round2_spectral.npz", "Evidence energy/slope diagnostic."),
    ("fig:r2-coverage-scale", "analysis/round2/plot_shortened_coverage.py", "analysis/round2/outputs/production_review/regional_scale_metrics.csv; analysis/round2/outputs/production/*_round2_fields.npz", "All-17 full-grid regional coverage versus scale."),
    ("fig:r2-full-burgers|fig:r2-full-bf|fig:r2-post-bm|fig:r2-post", "analysis/round2/plot_merged_family_curves.py", "analysis/round2/outputs/production_review/uncertainty_contrast_curves.csv", "Merged Burgers MLP/Fourier full/post-formation panels; aliases are one figure."),
    ("fig:r2-full-ac|fig:r2-full-af", "analysis/round2/plot_merged_family_curves.py", "analysis/round2/outputs/production_review/uncertainty_contrast_curves.csv", "Merged Allen–Cahn MLP/Fourier panels; aliases are one figure."),
]


def aux_labels(path):
    labels = {}
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"\\newlabel\s*\{([^{}]+)\}\s*\{", text):
            body, _ = group(text, match.end() - 1)
            values = []
            pos = 0
            while pos < len(body):
                if body[pos] == "{":
                    value, pos = group(body, pos)
                    values.append(value)
                else:
                    pos += 1
            if len(values) >= 2:
                labels[match[1]] = {"number": values[0], "page": values[1],
                                    "anchor": values[3] if len(values) > 3 else ""}
    return labels


def structural_objects(text, labels):
    objects = []
    counts = Counter()
    for match in FLOAT.finditer(text):
        env = match[1]
        endmatch = re.search(r"\\end\{" + re.escape(env) + r"\}", text[match.end():])
        if endmatch is None:
            raise ValueError(f"Unclosed float {env}")
        end = match.end() + endmatch.end()
        cursor = match.start()
        for caption in re.finditer(r"\\caption(?:\[[^]]*\])?\s*\{", text[match.end():end]):
            start = match.end() + caption.start()
            _, caption_end = group(text, match.end() + caption.end() - 1)
            tail_end = re.search(r"\\(?:begin|caption|end)\b", text[caption_end:end])
            label_end = caption_end + tail_end.start() if tail_end else end
            aliases = LABEL.findall(text[caption_end:label_end])
            if not aliases:
                aliases = LABEL.findall(text[cursor:caption_end])
            kind = "table" if env.startswith("table") else "figure"
            counts[kind] += 1
            anchor = next((labels[key] for key in aliases if key in labels), {})
            image = re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^{}]+)\}", text[cursor:caption_end])
            objects.append({"kind": kind, "number": anchor.get("number", str(counts[kind])),
                            "labels": aliases, "start": cursor, "end": label_end,
                            "page": anchor.get("page", ""), "image": "; ".join("paper/" + name for name in image)})
            cursor = label_end
    # Numbered equations are exact page anchors when the current aux is present.
    for match in re.finditer(r"\\begin\{(equation\*?|align\*?|gather\*?)\}", text):
        endmatch = re.search(r"\\end\{" + re.escape(match[1]) + r"\}", text[match.end():])
        if endmatch:
            end = match.end() + endmatch.end()
            aliases = LABEL.findall(text[match.start():end])
            anchor = next((labels[key] for key in aliases if key in labels), {})
            objects.append({"kind": "equation", "number": anchor.get("number", ""),
                            "labels": aliases, "start": match.start(), "end": end,
                            "page": anchor.get("page", ""), "image": ""})
    return objects


def section_events(text, labels):
    result = []
    section = subsection = subsubsection = 0
    appendix = False
    for match in re.finditer(r"\\appendix\b|\\(section|subsection|subsubsection)\*?\s*\{", text):
        if match[0] == r"\appendix":
            appendix = True
            section = subsection = subsubsection = 0
            continue
        title, end = group(text, match.end() - 1)
        names = LABEL.findall(text[end:end + 180].split("\n\n")[0])
        if match[1] == "section":
            section += 1
            subsection = subsubsection = 0
            fallback = chr(64 + section) if appendix else str(section)
        elif match[1] == "subsection":
            subsection += 1
            subsubsection = 0
            fallback = f"{chr(64 + section) if appendix else section}.{subsection}"
        else:
            subsubsection += 1
            fallback = f"{chr(64 + section) if appendix else section}.{subsection}.{subsubsection}"
        anchor = next((labels[key] for key in names if key in labels), {})
        result.append({"position": match.start(), "title": plain(title),
                       "label": names[0] if names else "", "number": anchor.get("number", fallback),
                       "page": anchor.get("page", "")})
    return result


def metadata_ranges(text):
    result = []
    names = (r"(?:label|(?:page|eq)?ref|cite\w*|tag|url|path|includegraphics|bibliography|bibliographystyle|"
             r"author|title|thanks|jmlrheading|ShortHeadings|setcounter|setlength|fontsize|"
             r"hspace|vspace|rule|texttt)")
    for match in re.finditer(r"\\" + names + r"\*?\b(?:\s*\[[^]]*\])*\s*\{", text):
        _, end = group(text, match.end() - 1)
        # Layout commands with a second brace argument can contain another digit.
        if re.match(r"\\(?:setcounter|setlength|fontsize|rule)", match[0]):
            pos = end
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos < len(text) and text[pos] == "{":
                _, end = group(text, pos)
        result.append((match.start(), end, "document/reference/layout/identifier argument"))
    for match in re.finditer(r"\\begin\{(?:tabular\*?|minipage)\}(?:\[[^]]*\])?\s*\{", text):
        _, end = group(text, match.end() - 1)
        result.append((match.start(), end, "table/layout specification"))
    return result


def classification(token, context, left, owner, source):
    value = plain(token)
    if owner and owner["kind"] == "equation":
        return "analytic", "Formula constant/exponent; not an empirical estimate."
    if source in {"oa_lean.tex", "j_theory.tex", "trim_aux_proofs.tex"}:
        return "analytic", "Formal-development statement/map; no floating-point result is certified by declaration presence."
    if re.search(r"(?:seed|seeds|index|rank|layer|features?|blocks?|slices?|rows?|steps?|modes?|nodes?|parameters?|points?|grid|float)\s*[$={~ ]*$", left[-45:], re.I):
        return "configuration", "Model/protocol/index token; inspect context for measured-versus-chosen counts."
    if re.search(r"(?:\^|_)\s*\{?\s*$", left[-12:]):
        return "analytic", "Mathematical exponent or index, not independently treated as empirical data."
    if re.search(r"(?:nominal|targeting|target|percentiles?|quantiles?|top-|threshold|cutoff|tolerance|set to|fixed at)\b", context, re.I) and value.rstrip("%") in {"0", "1", "2", "3", "4", "5", "10", "68", "70", "95", "0.68", "0.95", "1.96"}:
        return "configuration", "Candidate defined threshold/nominal target; occurrence-level semantic review remains visible."
    if owner and owner["kind"] == "table" and "&" not in context:
        return "numerical-result", "Table caption/note numerical claim or configured index; see context."
    if re.search(r"\b(?:proof|identity|counterexample|Pythagoras|inequality|witness)\b", context, re.I) or "10304" in token or "503" in token:
        return "analytic", "Explicit mathematical example or proof context."
    if source in {"j_setting.tex", "j_appendix.tex"} and not (owner and owner["kind"] == "table") and re.search(r"equation|trained networks|Adam|quadrature|uniform|Jacobian|percentile|tau|training|weight|width|bandwidth|learning rate|sub.network|initialis", context, re.I):
        return "configuration", "Setting/protocol declaration or associated numerical validation; classification requires semantic review."
    return "numerical-result", "Digit-bearing scientific occurrence; value-level producer/selector not automatically inferred."


def provenance(owner, context, source, label_map):
    if owner:
        for key in owner["labels"]:
            if key in label_map:
                return label_map[key], "output-level mapping"
    references = re.findall(r"\\(?:ref|eqref)\{([^{}]+)\}", context)
    for key in references:
        if key in label_map:
            return label_map[key], "nearby explicit output reference; scalar selector still requires review"
    if source in {"round3_math.tex", "round3_math_results.tex"}:
        return {"producer": "analysis/round3_local/review_round3.py",
                "source_artifact": "analysis/round3_local/outputs; analysis/theta_centred/outputs/review",
                "notes": "Saved-data bounds/decomposition; later direct theta computations are separately archived."}, "file-level evidence chain"
    if source == "short_seed_table.tex":
        return {"producer": "analysis/n1/summarize_n1.py; analysis/round3_local/review_round3.py",
                "source_artifact": "analysis/n1/outputs_cpu/*_n1.csv; analysis/round3_local/outputs/refined_roots.csv",
                "notes": "Saved core-family scales and sampled crossing brackets; literal selector still requires assignment."}, "file-level evidence chain"
    if source == "j_appendix.tex":
        return {"producer": "training/analysis/solve_gate_table.py; analysis/round3_local/review_round3.py; analysis/theta_centred/scripts/summarize_theta.py; legacy_core/verify_reference.py",
                "source_artifact": "provenance/training_configs.json; analysis/round3_local/outputs; analysis/theta_centred/outputs/review; original reference/control validation records",
                "notes": "Experimental appendix chain includes configurations, common-threshold subset and reference checks. Scalar selector/classification still require review."}, "file-level evidence chain"
    if source in {"j_round2_appendix.tex", "oa_empirical.tex", "oa_numerics.tex", "j_ceiling.tex", "j_abstract.tex", "j_intro.tex", "j_discussion.tex", "j_mechanism.tex", "j_offmanifold.tex", "j_setting.tex"}:
        routes = [
            (r"held.out|129|last.layer|98\.8|representab|feature express", "analysis/n5/n5_heldout.py; analysis/n5/summarise_n5.py", "analysis/n5/outputs; analysis/n5/input"),
            (r"ensemble|omission|leave.one|member", "analysis/round2/ensemble_loo.py; training/analysis/ensemble_calibration.py", "analysis/n5/input/burgers_mlp_seed*/*_fields.npz; analysis/round2/outputs/ensemble"),
            (r"conformal", "training/m6_conformal.py", "training/results; original raw covariance and recorded split inputs"),
            (r"theta|checkpoint.cent|nonlinear|prior.cent|prior.only", "analysis/theta_centred/scripts/summarize_theta.py; analysis/theta_centred/scripts/select_theta_evidence.py; analysis/round3_local/review_round3.py", "analysis/theta_centred/outputs/review; analysis/theta_centred/outputs/evidence; analysis/round3_local/outputs"),
            (r"evidence|perturb|slope|roundoff|synthetic|toy|backward|remainder|normwise", "analysis/round2/review_evidence.py; analysis/round2/review_production.py; analysis/round2/validate_round2.py; analysis/round2/stress_evidence_toy.py", "analysis/round2/outputs/production; analysis/round2/outputs/production_review; recorded synthetic-control outputs"),
            (r"root|crossing|bracket|envelope", "analysis/round3_local/review_round3.py; analysis/theta_centred/scripts/summarize_theta.py", "analysis/round3_local/outputs; analysis/theta_centred/outputs/review/root_comparisons.csv"),
            (r"admitt|attempt|reject|threshold|gate", "training/analysis/solve_gate_table.py; analysis/round3_local/review_round3.py", "provenance/training_configs.json; training/results; analysis/round3_local/outputs/common_threshold_subset.csv"),
            (r"contrast|coverage|RMS|precision|loading|spectr|sharp|bulk|region|scale", "analysis/round2/review_production.py; analysis/round2/report_saved_summaries.py", "analysis/round2/outputs/production; analysis/round2/outputs/production_review"),
        ]
        for pattern, producer, data in routes:
            if re.search(pattern, plain(context), re.I):
                return {"producer": producer, "source_artifact": data, "notes": "Candidate evidence chain inferred from local topic; no scalar equality was checked."}, "topic-level mapping; manual scalar selector pending"
        fallback = {
            "j_round2_appendix.tex": ("analysis/round2/review_production.py; analysis/round2/review_evidence.py; analysis/theta_centred/scripts/summarize_theta.py", "analysis/round2/outputs/production; analysis/round2/outputs/production_review; analysis/theta_centred/outputs/review"),
            "oa_numerics.tex": ("analysis/round2/review_production.py; analysis/round2/review_evidence.py; analysis/round2/validate_round2.py; analysis/round2/stress_evidence_toy.py; analysis/theta_centred/scripts/review_metadata.py", "analysis/round2/outputs; analysis/theta_centred/outputs/review; recorded synthetic-control outputs"),
            "j_mechanism.tex": ("analysis/round2/review_production.py; analysis/recompute_saved_summaries.py", "analysis/round2/cohort.json; analysis/round2/outputs/production_review; original spectral/loading JSON records"),
        }
        if source in fallback:
            producer, data = fallback[source]
            return {"producer": producer, "source_artifact": data,
                    "notes": "Documented file-level producer family only; the individual quantity selector remains unresolved."}, "file-level evidence chain; scalar selector pending"
    return None, "unresolved numerical meaning/producer"


def join_checked_claims(rows, paper_root, reproduced):
    """Join only explicit source-pattern spans and their registered token roles."""
    report_path = reproduced / "headline_claims/headline_claims.json" if reproduced else None
    if report_path is None or not report_path.is_file():
        return {"present": False, "linked_occurrences": 0, "checked_occurrences": 0}
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("schema") != "headline_claims/1":
        raise ValueError("Unsupported checked-claim report schema")
    roles = {
        "H01": {"17"}, "H02": {"two"}, "H03": {"eight"}, "H04": {"five"},
        "H05": {"four"}, "H06": {"three"}, "H07": {"eight"}, "H09": {"17"},
        "H10": {"68%"}, "H11": {"30.3"}, "H12": {"55.7%"},
        "H14": {"16/17"}, "H15": {"16/17"},
    }
    links = {}
    for claim in report.get("quantities", []):
        role = roles.get(claim["id"].split("_")[0], set())
        for registered in claim.get("occurrences", []):
            source = registered["source"]
            text = uncomment((paper_root / source).read_text(encoding="utf-8-sig"))
            for match in re.finditer(registered["pattern"], text, flags=re.MULTILINE):
                first_line = text.count("\n", 0, match.start()) + 1
                last_line = text.count("\n", 0, match.end()) + 1
                for row in rows:
                    if row["paper_source"] != "paper/" + source or not first_line <= row["source_line"] <= last_line:
                        continue
                    # These token roles are specific to the registered sentence;
                    # equality elsewhere in the manuscript does not create a link.
                    if row["readable_token"].replace(" ", "").lower() in role:
                        links.setdefault(row["occurrence_id"], []).append(claim)
    checked = 0
    for row in rows:
        claims = links.get(row["occurrence_id"], [])
        if not claims:
            continue
        row["verified_claim_ids"] = "; ".join(dict.fromkeys(claim["id"] for claim in claims))
        row["checked_report"] = "headline_claims/headline_claims.json (relative to the selected reproduction run)"
        row["checked_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        row["recomputed_value"] = json.dumps({claim["id"]: claim["computed_value"] for claim in claims})
        row["display_rule"] = "; ".join(json.dumps(claim["expected"]) for claim in claims)
        row["check_basis"] = "Explicit registered source-pattern span AND registered numerical-token role, not global numerical-string matching."
        row["quantity_id"] = row["verified_claim_ids"]
        row["quantity_description"] = "; ".join(claim["description"] for claim in claims)
        row['producer'] = 'tools/check_headline_claims.py; verification/headline_claims.json (underlying calculation and input selector)'
        artifacts = []
        for claim in claims:
            for record in claim.get('records', []):
                record_path = Path(record['path'])
                try:
                    location = 'reproduced/' + record_path.resolve().relative_to(reproduced.resolve()).as_posix()
                except ValueError:
                    location = record['path'].replace('\\', '/')
                    if '/analysis/' in location:
                        location = 'analysis/' + location.split('/analysis/', 1)[1]
                    elif '/training/' in location:
                        location = 'training/' + location.split('/training/', 1)[1]
                    else:
                        location = record_path.name
                artifacts.append(location)
        row['source_artifact'] = '; '.join(dict.fromkeys(artifacts))
        passed = all(claim.get("status") == "PASS" and claim.get("numeric_check") and claim.get("paper_check") for claim in claims)
        row["status"] = "CHECKED" if passed else "FAIL"
        row["display_match"] = "REGISTERED_CLAIM_PASS" if passed else "REGISTERED_CLAIM_FAIL"
        row["verification_level"] = "; ".join(dict.fromkeys(claim["verification_level"] for claim in claims))
        row["selector_status"] = "REGISTERED_CLAIM_SELECTOR"
        row["notes"] += " Linked checks apply only to the registered claim and their reported saved/fresh input level; they do not verify neighbouring numerical claims."
        checked += passed
    return {"present": True, "report_status": report.get("status"), "linked_occurrences": len(links),
            "checked_occurrences": checked, "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT / "paper")
    parser.add_argument("--out", type=Path, default=ROOT / "numerical_ledger")
    parser.add_argument("--output-map", type=Path, default=(ROOT / "docs/PAPER_OUTPUT_MAP.md")
                        if (ROOT / "docs/PAPER_OUTPUT_MAP.md").is_file() else ROOT / "repository/docs/PAPER_OUTPUT_MAP.md")
    parser.add_argument("--reproduced", type=Path, help="Join explicit registered claims from this saved-data reproduction run.")
    parser.add_argument("--page-index", type=Path, help="SyncTeX source-line page index; defaults to paper/page_index.json.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paper = Paper(args.paper_root.resolve())
    paper.read(paper.root / "full_paper.tex")
    paper.finish()
    text = paper.text
    begin = text.index(r"\begin{document}") + len(r"\begin{document}")
    aux_path = paper.root / "full_paper.aux"
    labels = aux_labels(aux_path)
    if not labels and (paper.root / 'label_index.json').is_file():
        labels = json.loads((paper.root / 'label_index.json').read_text(encoding='utf-8'))['labels']
    objects = structural_objects(text, labels)
    sections = section_events(text, labels)
    hidden = metadata_ranges(text)
    output_records = []
    label_map = {}
    for kind, collection in (("table", TABLES), ("figure", FIGURES)):
        for aliases, producer, artifact, notes in collection:
            found = next((obj for obj in objects if obj["kind"] == kind and set(obj["labels"]) & set(aliases.split("|"))), None)
            record = {"output_id": f"{kind}:{aliases.split('|')[0]}", "kind": kind,
                      "number": found["number"] if found else "", "labels": aliases,
                      "page": found["page"] if found else "", "page_basis": "AUX_FLOAT_ANCHOR" if found and found["page"] else "NOT_AVAILABLE",
                      "paper_source": "paper/" + paper.origin(found["start"]).source if found else "",
                      "source_line": paper.origin(found["start"]).line if found else "",
                      "asset": found["image"] if found else "", "producer": producer,
                      "source_artifact": artifact, "verification_status": "PROVENANCE_MAPPED" if found else "PENDING_OUTPUT_MATCH",
                      "verification_level": "DOCUMENTED_OUTPUT_CHAIN_ONLY", "notes": notes}
            output_records.append(record)
            for key in aliases.split("|"):
                label_map[key] = record

    occurrences = []
    excluded = []
    for match in number_tokens(text, begin):
        position = match.start()
        origin = paper.origin(position)
        metadata = next((why for start, end, why in hidden if start <= position < end), "")
        if metadata:
            excluded.append({"paper_source": "paper/" + origin.source, "source_line": origin.line,
                             "printed_token": match[0], "classification": "documentmetadata", "reason": metadata})
            continue
        paragraph_start = text.rfind("\n\n", begin, position)
        paragraph_start = max(begin, paragraph_start + 2)
        paragraph_end = text.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(text)
        context = text[max(paragraph_start, position - 250):min(paragraph_end, match.end() + 250)]
        line_start = text.rfind("\n", 0, position) + 1
        line_end = text.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        source_context = text[line_start:line_end].strip()
        owners = [obj for obj in objects if obj["start"] <= position < obj["end"]]
        owner = min(owners, key=lambda obj: obj["end"] - obj["start"]) if owners else None
        prior_sections = [section for section in sections if section["position"] < position]
        section = prior_sections[-1] if prior_sections else {"title": "Abstract / front matter", "label": "abstract", "number": "", "page": "1"}
        kind, class_note = classification(match[0], context, text[max(begin, position - 80):position], owner, origin.source)
        route, route_basis = provenance(owner, context, origin.source, label_map)
        if kind == "analytic":
            route = {"producer": "paper/" + origin.source, "source_artifact": "paper/" + origin.source,
                     "notes": "Analytic source statement/proof; no automatic claim of Lean coverage."}
            route_basis = "analytic source location"
        elif kind == "configuration" and route is None:
            route = {"producer": "paper/" + origin.source, "source_artifact": "paper/" + origin.source,
                     "notes": "Declared configuration in manuscript; independent config-file comparison pending."}
            route_basis = "declared configuration only"
        table_row = plain(source_context.split("&")[0]) if owner and owner["kind"] == "table" else ""
        column = source_context[:max(0, position - line_start)].count("&") + 1 if owner and owner["kind"] == "table" else ""
        macro_id = "macro:" + origin.macro if origin.macro else ""
        occurrence_id = f"O{len(occurrences) + 1:05d}"
        exact_page = owner["page"] if owner else ""
        occurrences.append({
            "occurrence_id": occurrence_id,
            "quantity_id": macro_id or f"occurrence:{origin.source}:{origin.line}:{len(occurrences) + 1}",
            "paper_source": "paper/" + origin.source, "source_line": origin.line,
            "source_sha256": paper.files[origin.source],
            "section_number": section["number"], "section_title": section["title"], "section_label": section["label"],
            "pdf_page": exact_page, "page_basis": "AUX_FLOAT_OR_EQUATION_ANCHOR" if exact_page else "NOT_RESOLVED_TO_PAGE",
            "section_anchor_page": section["page"],
            "table_or_figure": f"{owner['kind']} {owner['number']}" if owner and owner["kind"] in {"table", "figure"} else "",
            "object_labels": "|".join(owner["labels"]) if owner else "", "table_row": table_row, "table_column": column,
            "printed_token": match[0], "readable_token": plain(match[0]), "token_form": match.form,
            "word_count_value": COUNT_WORDS.get(match[0].lower(), "") if match.form == "RELEVANT_WORD_COUNT" else "",
            "classification": kind,
            "source_context": source_context, "context": plain(context),
            "macro_name": origin.macro, "macro_definition": origin.definition,
            "producer": route["producer"] if route else "", "source_artifact": route["source_artifact"] if route else "",
            "mapping_basis": route_basis, "recomputed_value": "", "display_rule": "PENDING_VALUE_SELECTOR_AND_ROUNDING_REVIEW",
            "display_match": "NOT_TESTED", "verification_level": "PROVENANCE_ONLY",
            "status": "PROVENANCE_MAPPED" if route else "PENDING_MAPPING",
            "notes": class_note + " " + (route["notes"] if route else "Numerical meaning and producer must be assigned explicitly.")
        })

    page_path = args.page_index or paper.root / 'page_index.json'
    page_data = json.loads(page_path.read_text(encoding='utf-8')) if page_path.is_file() else {}
    for row in occurrences:
        if row['pdf_page']:
            continue
        key = f"{row['paper_source']}:{row['source_line']}"
        source_key = row['paper_source'].removeprefix('paper/')
        if page_data.get('sources', {}).get(source_key) != row['source_sha256']:
            continue  # A stale line index must not give a current-page claim.
        if key in page_data.get('pages', {}):
            row['pdf_page'] = page_data['pages'][key]
            row['page_basis'] = 'SYNCTEX_UNIQUE_SOURCE_LINE_PAGE'
        elif key in page_data.get('candidates', {}):
            row['notes'] += ' Source-line candidate pages: ' + str(page_data['candidates'][key])
    checked_claims = join_checked_claims(occurrences, paper.root, args.reproduced)
    summary = {"scope": "Active-LaTeX digit-bearing scientific occurrence inventory; not a complete value verification.",
               "scientific_occurrences": len(occurrences), "excluded_metadata_tokens": len(excluded),
               "classifications": dict(Counter(row["classification"] for row in occurrences)),
               "token_forms": dict(Counter(row["token_form"] for row in occurrences)),
               "statuses": dict(Counter(row["status"] for row in occurrences)),
               "exact_float_or_equation_pages": sum(bool(row["pdf_page"]) for row in occurrences),
               "active_source_files": len(paper.files), "table_outputs": len(TABLES), "figure_outputs": len(FIGURES),
               "registered_claim_join": checked_claims,
               "unmatched_outputs": [row["output_id"] for row in output_records if row["verification_status"] != "PROVENANCE_MAPPED"],
               "source_hashes": {"paper/" + key: value for key, value in paper.files.items()},
               "aux_present": aux_path.is_file(), "aux_sha256": hashlib.sha256(aux_path.read_bytes()).hexdigest() if aux_path.is_file() else "",
               "output_map_present": args.output_map.is_file(),
               "output_map_sha256": hashlib.sha256(args.output_map.read_bytes()).hexdigest() if args.output_map.is_file() else "",
               "notes": ["Only explicit registered source-span/token-role joins to an executed numerical check can be labelled CHECKED.",
                         "File/topic provenance is weaker than a scalar-level calculation selector.",
                         "Every literal recurrence is retained; macros and registered claims may share quantity IDs.",
                         "Blank pdf_page means unknown; section_anchor_page is not claimed as the occurrence page.",
                         "Relevant verbal count patterns are included; other prose numerals and numeric labels drawn inside figure PDFs require manual review.",
                         "Generated numeric macro definitions are excluded as unprinted definitions, and expanded at every printed call site."]}
    write_csv(args.out / "occurrences.csv", occurrences)
    write_csv(args.out / "document_metadata.csv", excluded)
    write_csv(args.out / "table_figure_provenance.csv", output_records)
    (args.out / "occurrences.json").write_text(json.dumps({"summary": summary, "occurrences": occurrences}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "table_figure_provenance.json").write_text(json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "active_includes.json").write_text(json.dumps(paper.includes, indent=2), encoding="utf-8")
    doc = f"""# Numerical occurrence ledger: scope and status

This ledger inventories **{len(occurrences)} scientific numerical occurrences** in **{len(paper.files)} active TeX sources**, with repeated occurrences retained. It includes digit tokens and relevant spelled-out count patterns. It maps all **18 tables and six figures** to their documented producers and data. It does not assert that every value has been recomputed.

## How to rebuild

The reviewer entry point writes a fresh `ledger_inventory` directory automatically. To rebuild the index separately from the repository root, run `python tools/build_numerical_ledger.py --paper-root paper --out new_ledger --reproduced reproduced`. The supplied page and label indexes correspond to the released PDF; refresh them after a new LaTeX build or omit stale page coordinates. Paths in the records are relative to the public repository layout.

## What was automated

- Recursively expand active `\\input`/`\\include` commands starting at `full_paper.tex`; backup and inactive drafts are not scanned.
- Strip comments while preserving physical source-line coordinates.
- Suppress unprinted macro definitions and expand zero-argument numerical macros at each printed use. Each expanded value retains its call-site line and definition location.
- Inventory literal digit tokens, decimal/scientific expressions, powers of ten and count ratios, plus verbal counts near relevant scientific nouns (for example, five roots, eight seeds, two subspaces and one standard deviation). Include analytic exponents and declared settings rather than silently interpreting all numerals as empirical results.
- Associate each occurrence with its source line, current section, float/equation labels, and a candidate evidence chain. Caption aliases do not create extra tables or figures.
- Use current auxiliary-file page anchors for tables/figures and labelled equations, and unique SyncTeX source-line pages when supplied. Unresolved body-text pages remain blank; a section-start page is never represented as the actual occurrence page.

## What remains manual or separately computed

- Classification is a conservative lexical first pass. Configuration/analytic/result distinctions require semantic review, especially measured counts in setting paragraphs and nominal probabilities beside observed coverage.
- `PROVENANCE_MAPPED` means a documented producer/data chain exists. It is **not** `CHECKED`, `PASS` or agreement with raw data. Topic/file-level mappings are labelled explicitly.
- Each empirical occurrence needs a specific source selector, aggregation and rounding rule before a numerical checker can mark it checked. Unlinked quantities use occurrence-specific IDs instead of pretending equal printed values are the same measurement.
- `display_match=NOT_TESTED` and blank `recomputed_value` are intentional for provenance-only rows. With `--reproduced`, explicit registered claim spans and numerical-token roles are joined to their executed checks; no value is marked verified by matching its string to an arbitrary file.
- Analytic entries point to the manuscript statement/proof. The extractor does not infer Lean verification of each literal constant.

## Exact coverage boundaries

The inventory is a **lexical extraction with verbal-count patterns**, not a claim of exhaustive semantic extraction. Verbal numerals without a recognized nearby noun, numeric tick labels inside six figure assets, automatic equation/table/section numbering, bibliography metadata and numeric code identifiers are not scientific occurrence rows. Figure science is covered at the output/provenance level and requires its plot-data checks. Citation keys, cross-reference keys, document metadata and layout digits are separated in `document_metadata.csv`; bibliography files are not expanded.

Generated tables are included through the same active-input expansion as prose. A table interval may generate two endpoint rows; a scientific expression such as `1.94\\times10^{{-4}}` is one row. Source context retains the surrounding interval, inequality, units and shared powers. Table-column coordinates are source-cell hints, not a reconstructed multirow semantic schema.

The public interface should distinguish a completed numerical check from this provenance map. Missing data or unresolved claims must remain visible. The source hash/aux hash permits checking that the ledger corresponds to the released manuscript, and must be refreshed after pagination or text edits.

## Current inventory totals

- Classifications: `{json.dumps(summary['classifications'], ensure_ascii=False)}`.
- Statuses: `{json.dumps(summary['statuses'], ensure_ascii=False)}`.
- Occurrences with resolved source-line or float/equation page anchors: {summary['exact_float_or_equation_pages']}.
- Excluded document/reference/layout tokens: {len(excluded)}.
- Unmatched documented outputs: `{summary['unmatched_outputs']}`.
"""
    (args.out / "SCOPE_AND_STATUS.md").write_text(doc, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("scientific_occurrences", "classifications", "statuses", "active_source_files", "exact_float_or_equation_pages", "unmatched_outputs")}, indent=2))


if __name__ == "__main__":
    main()
