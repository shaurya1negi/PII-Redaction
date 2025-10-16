#Contains logic for PII redaction in standardized pdf , any text detected by PyMuPDF is routed here for PII analysis
import fitz  # PyMuPDF
import spacy
import re
from typing import List
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from .config import PIIConfig

config = PIIConfig()
# Use transformer model from config
nlp_configuration = config.NLP_CONFIG
provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
# spaCy fallback detector uses transformer model
nlp = spacy.load(config.NLP_CONFIG['models'][0]['model_name'])

#--------------------------------------------------------------------------------------------------------------------------------------------------------
def detect_pii_contextual(text: str) -> List[str]: # Contains the PII contextual detection logic on passed text
    """
    contextual PII detection - Microsoft Presidio.
    context and semantic is understood.
    """
    pii_entities = []
    
    try:
        # using presidio contextual analysis and validation
        results = analyzer.analyze(
            text=text,
            entities=[
			'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'IBAN_CODE', 'IP_ADDRESS', 'DATE_TIME', 'LOCATION',
            'IN_AADHAAR', 'IN_PAN', 'IN_VEHICLE_REGISTRATION', 'IN_PASSPORT'
            ],
            language='en'
        )
        
        # Eentities with confidence abobve 60% are taken (with contextual validation) 
        for result in results:
            if result.score >= 0.1:  # Only high - confidence detections
                entity_text = text[result.start:result.end].strip()
                
                # go through entity types while contextually filtering
                if _is_valid_pii_with_context(entity_text, result.entity_type, text, result.start, result.end):
                    pii_entities.append(entity_text)
        
        print(f"Presidio found {len(results)} entities, {len(pii_entities)} exceed high-confidence threshold")
    
    #Bellow exceptional handling might be redundant
    except Exception as e:
        print(f"Presidio failure, spaCy fallback: {e}")
        # revert to spaCy detection
        pii_entities = _fallback_spacy_detection(text)
    
    # include regualar expression pattern (useful for critical PII)
    critical_patterns = _detect_critical_patterns(text)
    pii_entities.extend(critical_patterns)
    
    return list(set(pii_entities))  # Filter dupes

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#Extra Layer of validation on detect_pii_contextual output to filter out common words but introduces computation overhead .

def _is_common_word(word: str) -> bool: 
    """Check if word is too common to be PII"""
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'age', 'any', 'big', 'end', 'far', 'got', 'let', 'put', 'run', 'top', 'try', 'use', 'way', 'win', 'yes', 'yet'
    }
    return word.lower() in common_words or len(word) <= 2

def _is_valid_pii_with_context(entity_text: str, entity_type: str, full_text: str, start: int, end: int) -> bool:
    """Enhanced contextual validation for different PII types"""
    entity_text = entity_text.strip()
    
    # common word check, basic length
    if len(entity_text) <= 1 or _is_common_word(entity_text):
        return False
    
    # get 50 characters  pre- and post- current object
    context_start = max(0, start - 30)
    context_end = min(len(full_text), end + 30)
    context = full_text[context_start:context_end].lower()
    
    if entity_type == "PERSON":
        # person validator
        if len(entity_text) <= 2:
            return False
        # presence of mixed case among multiple words
        if not (any(char.islower() for char in entity_text) or ' ' in entity_text):
            return False
        # skip if not a person by referencing indicators
        if any(indicator in context for indicator in ['.com', '.org', '.edu', 'http', 'www', '/', '\\']):
            return False
        return True
    
    elif entity_type == "ORGANIZATION":
        # non-common words (organizations)
        if len(entity_text) <= 4:
            return False
        # skipping generic words
        generic_orgs = {'company', 'corporation', 'inc', 'llc', 'ltd', 'department', 'office', 'center'}
        if entity_text.lower() in generic_orgs:
            return False
        return True
    
    elif entity_type == "LOCATION":
        # detect meaningful locations
        if len(entity_text) <= 2:
            return False
        # skipping directional and generic locations
        generic_locations = {'north', 'south', 'east', 'west', 'left', 'right', 'up', 'down', 'here', 'there'}
        if entity_text.lower() in generic_locations:
            return False
        return True
    
    elif entity_type == "DATE_TIME":
        return True
    
    elif entity_type == "URL":
        # URLs if detected, are sensitive
        return len(entity_text) > 5 and ('.' in entity_text or 'http' in entity_text.lower())
    
    elif entity_type == "IP_ADDRESS":
        # properly formatted IPs are also sensitive
        return len(entity_text) >= 7  # Minimum for x.x.x.x
    
    elif entity_type in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'IBAN_CODE', 'IP_ADDRESS', 'DATE_TIME', 'LOCATION',
            'IN_AADHAAR', 'IN_PAN', 'IN_VEHICLE_REGISTRATION', 'IN_PASSPORT']:
        # consider sensitive if passes confidence threshold
        return True
    
    # include if basic checks pass
    return True

#--------------------------------------------------------------------------------------------------------------------------------------------------------

def _fallback_spacy_detection(text: str) -> List[str]:
    """Enhanced spaCy-based detection as fallback - COMPREHENSIVE WITH CONTEXT"""
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entity_text = ent.text.strip()
        
        # names validator
        if ent.label_ == 'PERSON' and len(entity_text) > 2 and not _is_common_word(entity_text):
            if any(char.islower() for char in entity_text) or ' ' in entity_text:
                entities.append(entity_text)
        
        # organizations validator (context)
        elif ent.label_ == 'ORG' and len(entity_text) > 4:
            generic_orgs = {'company', 'corporation', 'inc', 'llc', 'ltd', 'department', 'office', 'center'}
            if entity_text.lower() not in generic_orgs:
                entities.append(entity_text)
        
        # location validator (specific, not gerneral)
        elif ent.label_ in ['GPE', 'LOC'] and len(entity_text) > 2:
            generic_locations = {'north', 'south', 'east', 'west', 'left', 'right', 'up', 'down', 'here', 'there'}
            if entity_text.lower() not in generic_locations:
                entities.append(entity_text)
        
        # dates with special patterns
        elif ent.label_ == 'DATE':
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\b\d{1,2}\s+\w+\s+\d{4}\b', entity_text):
                entities.append(entity_text)
    
    return entities

def _detect_critical_patterns(text: str) -> List[str]:
    """Detect critical PII patterns with regex - COMPREHENSIVE"""
    patterns = []
    
    # Email validator
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    patterns.extend(emails)
    
    # mobile number validator
    phones = re.findall(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b', text)
    patterns.extend(phones)
    
    # SSN (Americal SSNs)
    ssns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)
    patterns.extend(ssns)
    
    # CC Numbers (standard patterns)
    credit_cards = re.findall(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', text)
    patterns.extend(credit_cards)
    
    # URLs
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
    patterns.extend(urls)
    
    # IP address
    ip_addresses = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
    patterns.extend(ip_addresses)
    
    return patterns

def redact_pdf_preserve_layout(input_path: str, output_path: str) -> bool:
    """
    Advanced layout-preserving PDF redaction with contextual PII detection.
    """
    try:
        print(f"Detected PDF: {input_path}")
        
        # Open PDF
        doc = fitz.open(input_path)
        print(f"Number of Pages: {len(doc)}")
        
        total_redactions = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(f"\n Processing page {page_num + 1}...")
            
            # getting text from page
            text = page.get_text()
            
            # PII Detection (contextual analysis)
            pii_entities = detect_pii_contextual(text)
            print(f"   Found {len(pii_entities)} PII entities, at page {page_num + 1}")
            
            # Ffind and redact PII
            page_redactions = 0
            for entity in pii_entities:
                # look for page entity text
                text_instances = page.search_for(entity)
                
                for inst in text_instances:
                    # annotation for redaction
                    page.add_redact_annot(inst, fill=(0, 0, 0))  # Black fill
                    page_redactions += 1
            
            # appply all redactions on rage
            page.apply_redactions()
            print(f"   Applied {page_redactions} true redactions on page {page_num + 1}")
            total_redactions += page_redactions
        
        # Save the redacted PDF
        doc.save(output_path)
        doc.close()
        
        print(f"\n ADVANCED CONTEXTUAL REDACTION success")
        print(f"Output: {output_path}")
        print(f"applied redactions: {total_redactions}")
        
        return True
        
    except Exception as e:
        print(f"Failure in redacting: {e}")
        import traceback
        traceback.print_exc()
        return None

print("Context pipeline ready.")