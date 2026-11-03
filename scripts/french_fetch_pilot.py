"""
Fetch 10 French tax + 10 French social security cases from Conseil d'État open data.
Pilot run to verify data quality before scaling.
"""
import requests, zipfile, io, json, sys, os, re
from lxml import etree
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "french"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAX_KEYWORDS = [
    'impôt sur le revenu', 'impôt sur les sociétés', 'taxe sur la valeur ajoutée',
    'TVA', 'code général des impôts', 'contributions directes',
    'taxe foncière', 'taxe d\'habitation', 'plus-values',
    'bénéfices industriels', 'fiscal', 'fiscalité'
]

SOCIAL_KEYWORDS = [
    'sécurité sociale', 'aide sociale', 'RSA', 'revenu de solidarité active',
    'allocation adulte handicapé', 'AAH', 'MDPH',
    'aide personnalisée au logement', 'APL', 'protection sociale',
    'prestation sociale', 'code de l\'action sociale',
    'allocation de solidarité', 'handicap'
]

# Words that appear in both domains — exclude cases that match both
AMBIGUOUS_KEYWORDS = ['allocation', 'cotisation']


def clean_html(text):
    """Remove HTML tags, collapse whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_decision(xml_bytes):
    """Parse a Conseil d'État XML decision."""
    if not xml_bytes or len(xml_bytes.strip()) == 0:
        return None
    parser = etree.XMLParser(recover=True)
    try:
        root = etree.fromstring(xml_bytes, parser)
    except Exception:
        return None

    ti = root.find('.//Texte_Integral')
    raw_text = etree.tostring(ti, method='text', encoding='unicode') if ti is not None else ''
    html_text = etree.tostring(ti, encoding='unicode') if ti is not None else ''

    return {
        'id': root.findtext('.//Identification', ''),
        'court': root.findtext('.//Nom_Juridiction', '').strip(),
        'case_number': root.findtext('.//Numero_Dossier', ''),
        'date': root.findtext('.//Date_Lecture', ''),
        'ecli': root.findtext('.//Numero_ECLI', ''),
        'type_recours': root.findtext('.//Type_Recours', ''),
        'solution': root.findtext('.//Solution', ''),
        'formation': root.findtext('.//Formation_Jugement', ''),
        'text': clean_html(html_text),
        'text_length': len(raw_text),
    }


def classify_domain(text):
    """Classify a decision as 'tax', 'social', or None."""
    text_lower = text.lower()

    tax_score = sum(1 for kw in TAX_KEYWORDS if kw.lower() in text_lower)
    social_score = sum(1 for kw in SOCIAL_KEYWORDS if kw.lower() in text_lower)

    # Must have clear signal for one domain and not the other
    if tax_score >= 2 and social_score == 0:
        return 'tax'
    if social_score >= 2 and tax_score == 0:
        return 'social'
    return None


def fetch_month(year, month):
    """Download and parse one month of Conseil d'État decisions."""
    url = f'https://opendata.justice-administrative.fr/DCE/{year}/{month:02d}/CE_{year}{month:02d}.zip'
    print(f'Downloading {url}...')
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f'  Failed: {r.status_code}')
        return []

    z = zipfile.ZipFile(io.BytesIO(r.content))
    decisions = []
    for name in z.namelist():
        xml_bytes = z.read(name)
        dec = parse_decision(xml_bytes)
        if dec is None:
            continue
        dec['source_file'] = name
        decisions.append(dec)

    print(f'  Got {len(decisions)} decisions')
    return decisions


def main():
    target_tax = 200
    target_social = 200
    tax_cases = []
    social_cases = []

    # Try recent months until we have enough
    # All of 2024 + 2023 if needed
    months = [(2024, m) for m in range(12, 0, -1)] + [(2023, m) for m in range(12, 0, -1)]

    for year, month in months:
        if len(tax_cases) >= target_tax and len(social_cases) >= target_social:
            break

        decisions = fetch_month(year, month)

        for dec in decisions:
            domain = classify_domain(dec['text'])
            if domain == 'tax' and len(tax_cases) < target_tax:
                # Skip very short decisions (procedural)
                if dec['text_length'] < 3000:
                    continue
                dec['domain'] = 'tax'
                tax_cases.append(dec)
                print(f'  TAX [{len(tax_cases)}/{target_tax}]: {dec["case_number"]} ({dec["text_length"]} chars)')
            elif domain == 'social' and len(social_cases) < target_social:
                if dec['text_length'] < 3000:
                    continue
                dec['domain'] = 'social'
                social_cases.append(dec)
                print(f'  SOC [{len(social_cases)}/{target_social}]: {dec["case_number"]} ({dec["text_length"]} chars)')

    all_cases = tax_cases + social_cases
    print(f'\nCollected: {len(tax_cases)} tax, {len(social_cases)} social')

    # Save
    outfile = OUTPUT_DIR / 'french_200_cases.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f'Saved to {outfile}')

    # Print summary
    print('\n--- Tax cases ---')
    for c in tax_cases:
        print(f'  {c["case_number"]} | {c["date"]} | {c["solution"]} | {c["text_length"]} chars')
    print('\n--- Social cases ---')
    for c in social_cases:
        print(f'  {c["case_number"]} | {c["date"]} | {c["solution"]} | {c["text_length"]} chars')


if __name__ == '__main__':
    main()
