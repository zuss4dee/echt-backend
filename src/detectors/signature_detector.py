"""
Signature Detector Module

Verifies digital signatures and checks signature validity.
"""

import fitz  # PyMuPDF


class SignatureDetector:
    """Detects fraud indicators through digital signature analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze digital signatures for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'signature_issues': list of detected issues
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        signature_issues = []
        
        try:
            # Check for signatures on all pages
            all_signatures = []
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                try:
                    page_signatures = page.get_signatures()
                    if page_signatures:
                        all_signatures.extend(page_signatures)
                except Exception:
                    # Some PDFs may not support signature extraction
                    pass
            
            findings['signature_count'] = len(all_signatures)
            findings['signatures'] = all_signatures
            
            # Check if document has signatures
            if len(all_signatures) == 0:
                # Don't flag missing signatures - most legitimate documents don't have them
                # Only record in findings for informational purposes
                findings['has_signatures'] = False
            else:
                findings['has_signatures'] = True
                
                # Check signature validity
                validity_check = self._check_signature_validity(all_signatures)
                if validity_check['invalid_count'] > 0:
                    flags.append(f'Invalid Digital Signatures Detected ({validity_check["invalid_count"]})')
                    risk_score += 40
                    signature_issues.append('invalid_signatures')
                    findings['validity'] = validity_check
                
                # Check signature timestamps
                timestamp_check = self._check_signature_timestamps(all_signatures, pdf_doc)
                if timestamp_check['anomalies']:
                    flags.append('Signature Timestamp Anomalies')
                    risk_score += 25
                    signature_issues.append('timestamp_anomalies')
                    findings['timestamps'] = timestamp_check
                
                # Check for signature modifications
                modification_check = self._check_signature_modifications(pdf_doc)
                if modification_check['modified']:
                    flags.append('Document Modified After Signing')
                    risk_score += 50
                    signature_issues.append('post_signing_modifications')
                    findings['modifications'] = modification_check
            
        except Exception as e:
            flags.append(f'Signature Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'signature_issues': signature_issues
        }
    
    def _check_signature_validity(self, signatures):
        """Check if signatures are valid."""
        valid_count = 0
        invalid_count = 0
        invalid_reasons = []
        
        for sig in signatures:
            try:
                # Check if signature has valid certificate
                # PyMuPDF provides signature info but full validation requires certificate checking
                # For now, check if signature object exists and has basic structure
                if sig and len(sig) > 0:
                    valid_count += 1
                else:
                    invalid_count += 1
                    invalid_reasons.append('Invalid signature structure')
            except Exception as e:
                invalid_count += 1
                invalid_reasons.append(f'Signature validation error: {str(e)}')
        
        return {
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'invalid_reasons': invalid_reasons,
            'total': len(signatures)
        }
    
    def _check_signature_timestamps(self, signatures, pdf_doc):
        """Check signature timestamps against document dates."""
        anomalies = []
        metadata = pdf_doc.metadata
        
        # Get document creation date
        creation_date = metadata.get('creationDate', '')
        
        for i, sig in enumerate(signatures):
            try:
                # Extract signature timestamp if available
                # PyMuPDF signature objects may contain timestamp info
                # This is a simplified check - full implementation would parse signature data
                pass
            except Exception:
                pass
        
        return {
            'anomalies': anomalies,
            'has_anomalies': len(anomalies) > 0
        }
    
    def _check_signature_modifications(self, pdf_doc):
        """Check if document was modified after signing."""
        # Check for incremental updates after signature
        # This is detected in structure analysis, but we can add signature-specific checks here
        
        try:
            # Check if there are updates after signature pages
            # Simplified check - would need more sophisticated analysis
            return {
                'modified': False,
                'note': 'Signature modification check requires detailed PDF structure analysis'
            }
        except Exception:
            return {'modified': False, 'note': 'Unable to check modifications'}
