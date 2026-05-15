import requests
import json
import re
import time
import os
from bs4 import BeautifulSoup


# ─── Step 1: Get the list of UN member countries ───────────────────────────────

def get_un_member_countries():
    """
    Scrape the list of UN member states from Wikipedia.
    Returns a list of dicts: [{"name": "...", "wiki_url": "..."}, ...]
    """
    url = "https://en.wikipedia.org/wiki/Member_states_of_the_United_Nations"
    print(f"[INFO] Fetching UN member states list from:\n       {url}")

    resp = requests.get(url, headers={"User-Agent": "InfoboxCollector/1.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    countries = []


    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            
            th = row.find("th", attrs={"scope": "row"})
            if th:
                link = th.find("a", href=True)
                if link and link["href"].startswith("/wiki/"):
                    name = link.get_text(strip=True)
                    wiki_url = "https://en.wikipedia.org" + link["href"]
                    if name and not name.startswith("[") and len(name) > 1:
                        countries.append({"name": name, "wiki_url": wiki_url})
                    continue

            
            cells = row.find_all("td")
            if cells:
                link = cells[0].find("a", href=True)
                if link and link["href"].startswith("/wiki/"):
                    name = link.get_text(strip=True)
                    wiki_url = "https://en.wikipedia.org" + link["href"]
                    if name and not name.startswith("[") and len(name) > 1:
                        countries.append({"name": name, "wiki_url": wiki_url})

    
    seen = set()
    unique = []
    for c in countries:
        if c["wiki_url"] not in seen:
            seen.add(c["wiki_url"])
            unique.append(c)

    print(f"[INFO] Found {len(unique)} UN member countries.")
    return unique


# ─── Step 2: Extract an infobox from a country's Wikipedia page ────────────────

def extract_infobox_html(wiki_url):
    
    resp = requests.get(wiki_url, headers={"User-Agent": "InfoboxCollector/1.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    
    infobox = soup.find("table", class_="infobox")
    return infobox


# ─── Step 3: Convert infobox HTML table → structured JSON ─────────────────────

def clean_text(element):
    
    if element is None:
        return ""
    # Remove reference tags like [1], [2], etc.
    for sup in element.find_all("sup"):
        sup.decompose()
    # Remove hidden elements
    for hidden in element.find_all(style=re.compile(r"display\s*:\s*none")):
        hidden.decompose()
    text = element.get_text(separator=" ", strip=True)
    # Clean up residual bracket references
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_links(element):
    
    links = []
    if element is None:
        return links
    for a in element.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            href = "https://en.wikipedia.org" + href
        text = a.get_text(strip=True)
        if text and not text.startswith("["):
            links.append({"text": text, "href": href})
    return links


def infobox_to_json(infobox, country_name):
    
    if infobox is None:
        return {"country": country_name, "infobox_title": None, "data": [], "sections": []}

    result = {
        "country": country_name,
        "infobox_title": "",
        "data": [],
        "sections": []
    }

    # Get the infobox title
    title_row = infobox.find("th", class_=re.compile(r"infobox-above|fn"))
    if title_row:
        result["infobox_title"] = clean_text(title_row)

    # Process all rows
    rows = infobox.find_all("tr")
    current_section = None

    for row in rows:
        # Check if this is a section header (merged header row)
        header_th = row.find("th", class_=re.compile(r"infobox-header"))
        if header_th:
            current_section = {
                "header": clean_text(header_th),
                "rows": []
            }
            result["sections"].append(current_section)
            continue

        # Check for label-value pairs
        th = row.find("th", class_=re.compile(r"infobox-label"))
        td = row.find("td", class_=re.compile(r"infobox-data"))

        # Fallback: some infoboxes don't use standard classes
        if th is None:
            th = row.find("th")
        if td is None:
            td = row.find("td")

        if th and td:
            label = clean_text(th)
            value = clean_text(td)
            links = extract_links(td)

            if not label:
                continue

            entry = {
                "label": label,
                "value": value,
            }
            if links:
                entry["links"] = links

            # Handle nested lists/sublists in the td
            sub_items = []
            for li in td.find_all("li"):
                li_text = clean_text(li)
                if li_text:
                    sub_items.append(li_text)
            if sub_items:
                entry["sub_items"] = sub_items

            if current_section is not None:
                current_section["rows"].append(entry)
            else:
                result["data"].append(entry)

    return result


# ─── Step 4: Main pipeline ────────────────────────────────────────────────────

def main():
    output_dir = "infobox_data"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get all UN member countries
    countries = get_un_member_countries()

    if not countries:
        print("[ERROR] No countries found! The Wikipedia page structure may have changed.")
        print("        Try the debug command to inspect the page manually.")
        return

    all_infoboxes = {}
    failed = []

    # Step 2 & 3: Extract and convert each country's infobox
    for i, country in enumerate(countries, 1):
        name = country["name"]
        url = country["wiki_url"]
        print(f"[{i}/{len(countries)}] Processing: {name}")

        try:
            infobox = extract_infobox_html(url)

            if infobox is None:
                print(f"  ⚠ No infobox found for {name}")
                failed.append({"name": name, "reason": "no_infobox"})
                continue

            # Convert to JSON
            json_data = infobox_to_json(infobox, name)
            all_infoboxes[name] = json_data

            # Save individual country file
            safe_name = re.sub(r'[^\w\-]', '_', name)
            filepath = os.path.join(output_dir, f"{safe_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Extracted {len(json_data['data'])} top-level fields, "
                  f"{len(json_data['sections'])} sections")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append({"name": name, "reason": str(e)})

        
        time.sleep(1)

    # Save the combined dataset
    combined_path = os.path.join(output_dir, "_all_countries.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_infoboxes, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Combined dataset saved to: {combined_path}")

    # Save the list of failures
    if failed:
        fail_path = os.path.join(output_dir, "_failed.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f"[WARN] {len(failed)} countries failed. See: {fail_path}")

    print(f"\n{'='*60}")
    print(f"  DONE: {len(all_infoboxes)} infoboxes collected out of {len(countries)} countries")
    print(f"  Output directory: {output_dir}/")
    print(f"  Individual files: {output_dir}/<Country_Name>.json")
    print(f"  Combined file:    {output_dir}/_all_countries.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
