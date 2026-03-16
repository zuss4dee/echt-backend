"""
PDF Structure Detector Module

Detects anomalies in PDF internal structure that indicate manipulation.
"""

import fitz  # PyMuPDF


class PDFStructureDetector:
    """Detects fraud indicators through PDF structure analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze PDF structure for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'anomalies': list of detected anomalies
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        anomalies = []
        
        try:
            # Check for incremental updates
            # Note: Incremental updates are very common in legitimate documents
            # (e.g., printing from iOS, saving multiple times, etc.)
            # Only flag if combined with other strong indicators
            incremental_updates = self._check_incremental_updates(pdf_doc)
            if incremental_updates['has_updates']:
                # Don't flag as suspicious by itself - too common in legitimate docs
                # Store in findings for informational purposes only
                findings['incremental_updates'] = {
                    'count': incremental_updates['update_count'],
                    'description': 'PDF has been incrementally updated (common in legitimate documents)'
                }
            
            # Check PDF version consistency
            version_check = self._check_pdf_version(pdf_doc)
            if version_check['inconsistent']:
                flags.append('PDF Version Inconsistency')
                risk_score += 15
                anomalies.append('version_inconsistency')
                findings['version_info'] = version_check
            
            # Check cross-reference table
            xref_check = self._check_xref_table(pdf_doc)
            if xref_check['anomalies']:
                flags.append('Cross-Reference Table Anomalies')
                risk_score += 25
                anomalies.append('xref_anomalies')
                findings['xref_analysis'] = xref_check
            
            # Check object count
            object_check = self._check_object_count(pdf_doc)
            if object_check['suspicious']:
                # Only flag if extremely unusual (very high threshold)
                # Store in findings but don't flag unless combined with other indicators
                findings['object_analysis'] = object_check
                # Only flag if it's extremely suspicious (very high or very low)
                if object_check.get('objects_per_page', 0) > 500 or (object_check.get('objects_per_page', 0) < 3 and object_check.get('page_count', 0) > 1):
                    flags.append('Extremely Unusual Object Count')
                    risk_score += 15
                    anomalies.append('object_count')
            
            # Check linearization
            linearization_check = self._check_linearization(pdf_doc)
            if linearization_check['not_linearized']:
                findings['linearization'] = linearization_check
                # Non-linearized PDFs are common, not necessarily suspicious
            
        except Exception as e:
            flags.append(f'Structure Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'anomalies': anomalies
        }
    
    def _check_incremental_updates(self, pdf_doc):
        """Check if PDF has incremental updates."""
        try:
            # Check xref length - multiple xrefs indicate incremental updates
            xref_length = pdf_doc.xref_length()
            
            # Try to detect incremental updates by checking xref entries
            update_count = 0
            has_updates = False
            
            # Check if document has been saved incrementally
            # This is a heuristic - if xref has multiple sections, it's likely incremental
            try:
                # Access xref stream to check for updates
                trailer = pdf_doc.xref_get_key(-1, "Prev")
                if trailer[0] == 1:  # Prev key exists, indicating incremental update
                    has_updates = True
                    update_count = 1
            except:
                pass
            
            # Alternative check: compare page count with object count
            # Incrementally updated PDFs often have orphaned objects
            if xref_length > 0:
                page_count = len(pdf_doc)
                # If object count is significantly higher than expected, may indicate updates
                if xref_length > page_count * 50:  # Heuristic threshold
                    has_updates = True
                    update_count = max(update_count, 1)
            
            return {
                'has_updates': has_updates,
                'update_count': update_count,
                'xref_length': xref_length
            }
        except Exception:
            return {'has_updates': False, 'update_count': 0, 'xref_length': 0}
    
    def _check_pdf_version(self, pdf_doc):
        """Check PDF version for inconsistencies."""
        try:
            pdf_version = pdf_doc.pdf_version()
            metadata = pdf_doc.metadata
            
            # Check if version matches expected for creation software
            producer = (metadata.get('producer', '') or '').lower()
            
            inconsistent = False
            reason = None
            
            # Very old PDF version with modern software = suspicious
            if pdf_version < 1.4 and ('acrobat' in producer or 'adobe' in producer):
                inconsistent = True
                reason = f'PDF version {pdf_version} is unusually old for Adobe software'
            
            # Very new PDF version with old software = suspicious
            if pdf_version >= 2.0 and ('word' in producer or 'excel' in producer):
                inconsistent = True
                reason = f'PDF version {pdf_version} is unusually new for Microsoft Office'
            
            return {
                'inconsistent': inconsistent,
                'version': pdf_version,
                'reason': reason,
                'producer': metadata.get('producer', 'Unknown')
            }
        except Exception:
            return {'inconsistent': False, 'version': 0, 'reason': None}
    
    def _check_xref_table(self, pdf_doc):
        """Check cross-reference table for anomalies."""
        anomalies = []
        try:
            xref_length = pdf_doc.xref_length()
            
            if xref_length == 0:
                anomalies.append('Empty xref table')
            
            # Check for corrupted xref entries
            # This is a basic check - more sophisticated analysis would check each entry
            try:
                # Try to access xref entries
                test_key = pdf_doc.xref_get_key(1, "Type")
                if test_key[0] == -1:  # Key not found or error
                    anomalies.append('Corrupted xref entries detected')
            except:
                anomalies.append('Unable to verify xref integrity')
            
            return {
                'anomalies': anomalies,
                'xref_length': xref_length,
                'status': 'suspicious' if anomalies else 'normal'
            }
        except Exception as e:
            return {
                'anomalies': [f'Xref analysis error: {str(e)}'],
                'xref_length': 0,
                'status': 'error'
            }
    
    def _check_object_count(self, pdf_doc):
        """Check for unusual object counts."""
        try:
            xref_length = pdf_doc.xref_length()
            page_count = len(pdf_doc)
            
            suspicious = False
            reason = None
            
            if page_count > 0:
                # Calculate objects per page
                objects_per_page = xref_length / page_count
                
                # Very high object count per page may indicate manipulation
                if objects_per_page > 200:
                    suspicious = True
                    reason = f'Unusually high object count: {objects_per_page:.1f} objects per page'
                
                # Very low object count may indicate stripped PDF
                if objects_per_page < 5 and page_count > 1:
                    suspicious = True
                    reason = f'Unusually low object count: {objects_per_page:.1f} objects per page'
            
            return {
                'suspicious': suspicious,
                'total_objects': xref_length,
                'page_count': page_count,
                'objects_per_page': objects_per_page if page_count > 0 else 0,
                'reason': reason
            }
        except Exception:
            return {'suspicious': False, 'total_objects': 0, 'page_count': 0}
    
    def _check_linearization(self, pdf_doc):
        """Check if PDF is linearized."""
        try:
            # PyMuPDF doesn't have direct linearization check
            # We can infer from document properties
            is_linearized = False
            
            # Check if document loads quickly (heuristic)
            # Linearized PDFs load faster, but this is not definitive
            
            return {
                'not_linearized': not is_linearized,
                'note': 'Linearization status cannot be definitively determined'
            }
        except Exception:
            return {'not_linearized': True, 'note': 'Unable to check linearization'}
