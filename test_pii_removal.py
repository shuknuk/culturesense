"""
Unit tests for PII/PHI removal layer.
Ensures patient data is scrubbed before processing.
"""

import unittest

from pii_removal import detect_pii, scrub_pii


class TestNameRemoval(unittest.TestCase):
    """Test patient name removal patterns."""

    def test_patient_name_label(self):
        """Standard Patient Name: label format."""
        text = "Patient Name: John Smith\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("John Smith", result)
        self.assertIn("[REDACTED NAME]", result)
        self.assertIn("Organism: E. coli", result)

    def test_patient_name_alt_label(self):
        """Patient: label format."""
        text = "Patient: Jane Doe\nDate: 2024-01-15"
        result = scrub_pii(text)
        self.assertNotIn("Jane Doe", result)
        self.assertIn("[REDACTED NAME]", result)

    def test_pt_name_label(self):
        """Pt Name: abbreviated label."""
        text = "Pt Name: Robert Johnson"
        result = scrub_pii(text)
        self.assertNotIn("Robert Johnson", result)
        self.assertIn("[REDACTED NAME]", result)

    def test_full_name_variations(self):
        """Various name formats - first last, last first, with comma."""
        test_cases = [
            ("Patient Name: Sarah Connor", "Sarah Connor"),
            ("Patient: Connor, Sarah", "Connor, Sarah"),
            ("Pt: J. Edgar Hoover", "J. Edgar Hoover"),
            ("Name: Mary-Jane Watson", "Mary-Jane Watson"),
        ]
        for text, name in test_cases:
            with self.subTest(text=text):
                result = scrub_pii(text)
                self.assertNotIn(name, result)
                self.assertIn("[REDACTED NAME]", result)


class TestDOBRemoval(unittest.TestCase):
    """Test date of birth removal patterns."""

    def test_dob_standard(self):
        """Standard DOB: label."""
        text = "DOB: 01/15/1980\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("01/15/1980", result)
        self.assertIn("[REDACTED DOB]", result)
        self.assertIn("Organism: E. coli", result)

    def test_date_of_birth_label(self):
        """Date of Birth: label."""
        text = "Date of Birth: 1980-01-15"
        result = scrub_pii(text)
        self.assertNotIn("1980-01-15", result)
        self.assertIn("[REDACTED DOB]", result)

    def test_dob_alt_formats(self):
        """Various DOB formats."""
        test_cases = [
            ("DOB: 01/15/1980", "01/15/1980"),
            ("DOB: 15/01/1980", "15/01/1980"),
            ("DOB: 01-15-1980", "01-15-1980"),
            ("DOB: Jan 15, 1980", "Jan 15, 1980"),
            ("DOB: January 15, 1980", "January 15, 1980"),
        ]
        for text, dob in test_cases:
            with self.subTest(text=text):
                result = scrub_pii(text)
                self.assertNotIn(dob, result)
                self.assertIn("[REDACTED DOB]", result)


class TestMRNRemoval(unittest.TestCase):
    """Test medical record number removal."""

    def test_mrn_standard(self):
        """Standard MRN: label."""
        text = "MRN: 12345678\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("12345678", result)
        self.assertIn("[REDACTED MRN]", result)

    def test_medical_record_number(self):
        """Full Medical Record Number label."""
        text = "Medical Record Number: ABC123456"
        result = scrub_pii(text)
        self.assertNotIn("ABC123456", result)
        self.assertIn("[REDACTED MRN]", result)

    def test_mr_number(self):
        """MR #: label format."""
        text = "MR #: 87654321"
        result = scrub_pii(text)
        self.assertNotIn("87654321", result)
        self.assertIn("[REDACTED MRN]", result)

    def test_account_number(self):
        """Account #: label."""
        text = "Account #: ACC987654321"
        result = scrub_pii(text)
        self.assertNotIn("ACC987654321", result)
        self.assertIn("[REDACTED MRN]", result)

    def test_patient_id(self):
        """Patient ID: label."""
        text = "Patient ID: PID12345"
        result = scrub_pii(text)
        self.assertNotIn("PID12345", result)
        self.assertIn("[REDACTED MRN]", result)


class TestSSNRemoval(unittest.TestCase):
    """Test SSN removal."""

    def test_ssn_with_dashes(self):
        """SSN with dashes format."""
        text = "SSN: 123-45-6789\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("123-45-6789", result)
        self.assertIn("[REDACTED SSN]", result)

    def test_social_security_number(self):
        """Full Social Security Number label."""
        text = "Social Security Number: 987654321"
        result = scrub_pii(text)
        self.assertNotIn("987654321", result)
        self.assertIn("[REDACTED SSN]", result)

    def test_ssn_plain(self):
        """Plain 9-digit number after SSN:."""
        text = "SSN: 555112222"
        result = scrub_pii(text)
        self.assertNotIn("555112222", result)
        self.assertIn("[REDACTED SSN]", result)


class TestPhoneRemoval(unittest.TestCase):
    """Test phone number removal."""

    def test_phone_standard(self):
        """Standard phone format."""
        text = "Phone: (555) 123-4567\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("(555) 123-4567", result)
        self.assertIn("[REDACTED PHONE]", result)

    def test_phone_with_dots(self):
        """Phone with dot separators."""
        text = "Phone: 555.123.4567"
        result = scrub_pii(text)
        self.assertNotIn("555.123.4567", result)
        self.assertIn("[REDACTED PHONE]", result)

    def test_phone_plain(self):
        """Plain phone format."""
        text = "Phone: 555-123-4567"
        result = scrub_pii(text)
        self.assertNotIn("555-123-4567", result)
        self.assertIn("[REDACTED PHONE]", result)


class TestEmailRemoval(unittest.TestCase):
    """Test email address removal."""

    def test_email_standard(self):
        """Standard email format."""
        text = "Email: patient@email.com\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("patient@email.com", result)
        self.assertIn("[REDACTED EMAIL]", result)

    def test_email_with_plus(self):
        """Email with plus sign."""
        text = "Email: patient.name+tag@hospital.org"
        result = scrub_pii(text)
        self.assertNotIn("patient.name+tag@hospital.org", result)
        self.assertIn("[REDACTED EMAIL]", result)


class TestAddressRemoval(unittest.TestCase):
    """Test address removal."""

    def test_address_standard(self):
        """Standard address format."""
        text = "Address: 123 Main Street, Springfield, IL 62701"
        result = scrub_pii(text)
        self.assertNotIn("123 Main Street", result)
        self.assertIn("[REDACTED ADDRESS]", result)

    def test_street_address(self):
        """Street Address: label."""
        text = "Street Address: 456 Oak Ave, Unit 2B"
        result = scrub_pii(text)
        self.assertNotIn("456 Oak Ave", result)
        self.assertIn("[REDACTED ADDRESS]", result)


class TestProviderRemoval(unittest.TestCase):
    """Test provider name removal."""

    def test_provider_standard(self):
        """Standard Provider: label."""
        text = "Provider: Dr. Sarah Chen\nOrganism: E. coli"
        result = scrub_pii(text, remove_provider_names=True)
        self.assertNotIn("Dr. Sarah Chen", result)
        self.assertIn("[REDACTED PROVIDER]", result)

    def test_physician_label(self):
        """Physician: label."""
        text = "Physician: Dr. Robert Smith"
        result = scrub_pii(text, remove_provider_names=True)
        self.assertNotIn("Dr. Robert Smith", result)
        self.assertIn("[REDACTED PROVIDER]", result)

    def test_provider_disabled(self):
        """Provider names kept when flag is False."""
        text = "Provider: Dr. Sarah Chen\nOrganism: E. coli"
        result = scrub_pii(text, remove_provider_names=False)
        self.assertIn("Dr. Sarah Chen", result)
        self.assertNotIn("[REDACTED PROVIDER]", result)


class TestCombinedPII(unittest.TestCase):
    """Test combined PII scenarios."""

    def test_full_patient_header(self):
        """Complete patient header block."""
        text = """Patient Name: John Smith
DOB: 01/15/1980
MRN: 12345678
SSN: 123-45-6789
Phone: (555) 123-4567
Email: john.smith@email.com
Address: 123 Main St, Springfield, IL

Organism: E. coli
CFU/mL: 100,000"""

        result = scrub_pii(text)

        self.assertNotIn("John Smith", result)
        self.assertNotIn("01/15/1980", result)
        self.assertNotIn("12345678", result)
        self.assertNotIn("123-45-6789", result)
        self.assertNotIn("(555) 123-4567", result)
        self.assertNotIn("john.smith@email.com", result)
        self.assertNotIn("123 Main St", result)

        self.assertIn("[REDACTED NAME]", result)
        self.assertIn("[REDACTED DOB]", result)
        self.assertIn("[REDACTED MRN]", result)
        self.assertIn("[REDACTED SSN]", result)
        self.assertIn("[REDACTED PHONE]", result)
        self.assertIn("[REDACTED EMAIL]", result)
        self.assertIn("[REDACTED ADDRESS]", result)

        # Medical data preserved
        self.assertIn("Organism: E. coli", result)
        self.assertIn("CFU/mL: 100,000", result)

    def test_preserve_medical_data(self):
        """Ensure medical data is not over-scrubbed."""
        text = """Patient Name: Jane Doe
MRN: 12345
Date: 2024-01-15

Organism: E. coli
CFU/mL: 50,000"""

        result = scrub_pii(text)

        # PII removed
        self.assertNotIn("Jane Doe", result)
        self.assertNotIn("12345", result)

        # Medical data preserved
        self.assertIn("Organism: E. coli", result)
        self.assertIn("CFU/mL: 50,000", result)
        self.assertIn("Date: 2024-01-15", result)  # Collection date, not DOB


class TestDetectPII(unittest.TestCase):
    """Test PII detection reporting."""

    def test_detect_name(self):
        """Detect patient name presence."""
        text = "Patient Name: John Smith\nOrganism: E. coli"
        detected = detect_pii(text)
        self.assertIn("name", detected)

    def test_detect_dob(self):
        """Detect DOB presence."""
        text = "DOB: 01/15/1980"
        detected = detect_pii(text)
        self.assertIn("dob", detected)

    def test_detect_mrn(self):
        """Detect MRN presence."""
        text = "MRN: 12345678"
        detected = detect_pii(text)
        self.assertIn("mrn", detected)

    def test_detect_none(self):
        """No PII detected."""
        text = "Organism: E. coli\nCFU/mL: 100,000"
        detected = detect_pii(text)
        self.assertEqual(detected, [])

    def test_detect_multiple(self):
        """Multiple PII types detected."""
        text = "Patient Name: John Smith\nDOB: 01/15/1980\nMRN: 12345"
        detected = detect_pii(text)
        self.assertIn("name", detected)
        self.assertIn("dob", detected)
        self.assertIn("mrn", detected)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and false positives."""

    def test_empty_text(self):
        """Handle empty string."""
        result = scrub_pii("")
        self.assertEqual(result, "")

    def test_no_pii(self):
        """Text with no PII passes through unchanged."""
        text = "Organism: E. coli\nCFU/mL: 100,000"
        result = scrub_pii(text)
        self.assertEqual(result, text)

    def test_partial_pii(self):
        """Text with some PII fields missing."""
        text = "Patient Name: John Smith\nOrganism: E. coli"
        result = scrub_pii(text)
        self.assertNotIn("John Smith", result)
        self.assertIn("Organism: E. coli", result)

    def test_multiline_address(self):
        """Multi-line address block."""
        text = """Address: 123 Main Street
Springfield, IL 62701

Organism: E. coli"""
        result = scrub_pii(text)
        self.assertNotIn("123 Main Street", result)
        self.assertIn("[REDACTED ADDRESS]", result)


if __name__ == "__main__":
    unittest.main()
