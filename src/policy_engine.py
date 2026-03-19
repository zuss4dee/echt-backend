import re

try:
    import yaml
except ImportError:
    yaml = None  # Install PyYAML for policy config: pip install PyYAML

_DEFAULT_CONFIG = {
    'global_thresholds': {'high_risk_score': 55, 'medium_risk_score': 35},
    'document_types': {}
}


class PolicyEngine:
    """
    Evaluates forensic analysis results against defined business policies.
    """

    def __init__(self, config_path='config/policies.yaml'):
        if yaml is None:
            self.config = _DEFAULT_CONFIG
            return
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = _DEFAULT_CONFIG
        if not self.config:
            self.config = _DEFAULT_CONFIG

    def evaluate(self, analysis, doc_type_key):
        """
        Apply policy rules to analysis results.

        Args:
            analysis: The dictionary returned by analyze_file()
            doc_type_key: Key matching a section in policies.yaml (e.g., 'visa_refusal')

        Returns:
            dict: {
                'verdict': 'RED'|'AMBER'|'GREEN',
                'reason': str,
                'action': str,
                'critical_flags_hit': list
            }
        """
        # 1. Get Policy for this Document Type
        policies = self.config.get('document_types', {})
        policy = policies.get(doc_type_key, {})

        # Defaults if policy not found
        if not policy:
            return self._evaluate_generic(analysis)

        # 2. Check "Auto-Reject" Rules
        # These are instant fail conditions defined in YAML
        auto_reject_rules = policy.get('auto_reject_if', [])
        for rule in auto_reject_rules:
            reject_reason = self._check_auto_reject(analysis, rule)
            if reject_reason:
                return {
                    'verdict': 'RED',
                    'reason': f"Policy Violation: {reject_reason}",
                    'action': "Reject immediately. Do not accept this document.",
                    'critical_flags_hit': [reject_reason]
                }

        # 3. Check Critical Flags (include all detectors: metadata, noise, structure, font, etc.)
        critical_flags_config = policy.get('critical_flags', [])
        hit_critical_flags = []

        # Flatten all detected flags from metadata, noise, and PDF detectors
        detected_flags = list(analysis['metadata'].get('flags', [])) + \
                        list(analysis['noise'].get('flags', []))
        all_results = analysis.get('all_results', {})
        for key in ('structure', 'font', 'text_layer', 'layout', 'signature', 'embedded'):
            if key in all_results and all_results[key].get('flags'):
                detected_flags.extend(all_results[key]['flags'])

        # Critical config can use short names; map to flag substrings where needed
        critical_substrings = {
            'pixel_manipulation': ('smoothing', 'manipulation', 'ela', 'noise'),
            'timeline_anomaly': ('timeline', 'time gap', 'temporal', 'modified after', 'anomaly'),
            'suspicious_software': ('photoshop', 'gimp', 'canva', 'digital manipulation software'),
        }

        for flag in detected_flags:
            flag_lower = flag.lower()
            for critical in critical_flags_config:
                c = critical.lower()
                if c in flag_lower:
                    hit_critical_flags.append(flag)
                    break
                elif critical == "author_is_person" and "author field" in flag_lower:
                    hit_critical_flags.append(flag)
                    break
                # Use substring mapping for known critical types
                for sub in critical_substrings.get(critical, ()):
                    if sub in flag_lower:
                        hit_critical_flags.append(flag)
                        break
                else:
                    continue
                break

        if hit_critical_flags:
            return {
                'verdict': 'RED',
                'reason': f"Critical Irregularities detected: {hit_critical_flags[0]}",
                'action': "Escalate to manager. High probability of forgery.",
                'critical_flags_hit': hit_critical_flags
            }

        # 4. Check Risk Scores (aggregate metadata + PDF detectors for better fraud detection)
        risk_score = analysis['metadata']['risk_score']
        trust_score = analysis['metadata']['trust_score']

        # Include PDF detector risk so structure/font/signature anomalies push toward RED
        all_results = analysis.get('all_results', {})
        detector_risks = []
        for key in ('structure', 'font', 'text_layer', 'layout', 'signature', 'embedded'):
            if key in all_results and all_results[key] is not None:
                r = all_results[key].get('risk_score')
                if r is not None:
                    detector_risks.append(r)
        max_detector_risk = max(detector_risks) if detector_risks else 0
        # Effective risk: metadata risk boosted by any high detector risk (so fraud indicators = RED)
        effective_risk = min(100, risk_score + 0.4 * max_detector_risk)

        # Adjust score based on weights if defined
        weights = policy.get('weights', {})
        if 'ai_content' in weights and analysis.get('ai_content', {}).get('is_ai_generated'):
            effective_risk = min(100, effective_risk * weights['ai_content'])

        thresholds = self.config.get('global_thresholds', {})
        high_risk = thresholds.get('high_risk_score', 55)
        medium_risk = thresholds.get('medium_risk_score', 35)

        if effective_risk >= high_risk or trust_score < 45:
            return {
                'verdict': 'RED',
                'reason': f"High Risk Score ({int(effective_risk)}/100)",
                'action': "Reject or request original digital evidence.",
                'critical_flags_hit': []
            }
        elif effective_risk >= medium_risk:
            return {
                'verdict': 'AMBER',
                'reason': "Moderate Anomalies Detected",
                'action': "Manual review required. Verify with issuing institution if possible.",
                'critical_flags_hit': []
            }
        else:
            return {
                'verdict': 'GREEN',
                'reason': "Passes Forensic Checks",
                'action': "Accept as evidence.",
                'critical_flags_hit': []
            }

    def _check_auto_reject(self, analysis, rule):
        """
        Parses simple string rules like 'producer_contains: aspose' or 'ai_confidence > 80'
        """
        rule = rule.lower()

        # Producer/Software Checks
        if "producer_contains" in rule or "aspose" in rule or "ilovepdf" in rule:
            target = rule.split(":")[-1].strip() if ":" in rule else rule
            raw = analysis['metadata'].get('raw_data', {})
            pdf_meta = raw.get('pdf_metadata', {})
            producer = str(raw.get('producer', '') or pdf_meta.get('producer', '')).lower()
            creator = str(raw.get('creator', '') or pdf_meta.get('creator', '')).lower()
            software = str(raw.get('software', '')).lower()
            if target in producer or target in creator or target in software:
                return f"Banned Software Detected: {target.title()}"

        # AI Confidence Checks
        if "ai_confidence" in rule:
            # Parse "ai_confidence > 80"
            match = re.search(r'ai_confidence\s*>\s*(\d+)', rule)
            if match:
                threshold = int(match.group(1))
                ai_result = analysis.get('ai_content', {})
                if ai_result and ai_result.get('confidence', 0) > threshold:
                    return f"AI Confidence > {threshold}%"

        return None

    def _evaluate_generic(self, analysis):
        """Fallback for unknown document types"""
        risk = analysis['metadata']['risk_score']
        trust = analysis['metadata'].get('trust_score', 85)
        all_results = analysis.get('all_results', {})
        max_d = 0
        for key in ('structure', 'font', 'signature', 'text_layer', 'layout', 'embedded'):
            if key in all_results and all_results[key]:
                r = all_results[key].get('risk_score') or 0
                max_d = max(max_d, r)
        effective = min(100, risk + 0.4 * max_d)
        if effective >= 55 or trust < 45:
            return {'verdict': 'RED', 'reason': 'Generic High Risk', 'action': 'Reject', 'critical_flags_hit': []}
        elif effective >= 35:
            return {'verdict': 'AMBER', 'reason': 'Generic Moderate Risk', 'action': 'Review', 'critical_flags_hit': []}
        return {'verdict': 'GREEN', 'reason': 'Low Risk', 'action': 'Accept', 'critical_flags_hit': []}
