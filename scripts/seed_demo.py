"""Load demo clients and documents through the public API (stdlib only).

Usage: python scripts/seed_demo.py [BASE_URL] [API_KEY]
"""

import json
import sys
import urllib.error
import urllib.request

CLIENTS = [
    {
        "client": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@neviswealth.com",
            "description": "Tech founder, interested in sustainable investing.",
            "social_links": ["https://www.linkedin.com/in/johndoe"],
        },
        "documents": [
            (
                "Utility bill - March 2026",
                "Electricity utility bill issued to John Doe, 12 Baker Street, London. "
                "Billing period March 2026. Amount due: 84.20 GBP.",
            ),
            (
                "Passport copy",
                "Scanned copy of passport. Nationality: British. Date of birth 01.02.1980. "
                "Passport number 123456789, expires 2031.",
            ),
            (
                "Investment policy statement",
                "Moderate risk tolerance, ten year horizon, 60/40 allocation between equities "
                "and bonds. The client accepts drawdowns of up to 15 percent.",
            ),
        ],
    },
    {
        "client": {
            "first_name": "Maria",
            "last_name": "Rossi",
            "email": "maria@rossi-family.it",
            "description": "Retired surgeon with a conservative risk profile; focused on estate "
            "planning for her grandchildren.",
        },
        "documents": [
            (
                "Tenancy agreement",
                "Residential tenancy agreement between the landlord and tenant Maria Rossi for "
                "the apartment at Via Roma 5, Milan, starting January 2025.",
            ),
            (
                "Last will and testament",
                "I, Maria Rossi, leave my property in Tuscany to my grandchildren in equal "
                "shares. Executor: Luca Rossi.",
            ),
            (
                "Tax return 2025",
                "Annual income tax return with capital gains of 12,000 EUR and dividend income "
                "of 3,400 EUR. Total tax paid: 5,100 EUR.",
            ),
        ],
    },
    {
        "client": {
            "first_name": "Ahmed",
            "last_name": "Al-Sayed",
            "email": "ahmed@alsayed.fr",
            "description": "Entrepreneur who sold his logistics company in 2024.",
        },
        "documents": [
            (
                "Bank statement",
                "Current account statement for September 2026 sent to 7 Rue de Rivoli, Paris. "
                "Closing balance 12,400 EUR.",
            ),
            (
                "Meeting notes",
                "Discussed funding university education for two children and buying a holiday "
                "home in Portugal within five years.",
            ),
        ],
    },
]


def post(base_url: str, path: str, payload: dict[str, object], api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        base_url + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> None:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    for entry in CLIENTS:
        try:
            client = post(base_url, "/clients", entry["client"], api_key)
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                print(f"skip {entry['client']['email']}: already exists")
                continue
            raise
        print(f"client {client['id']} {client['email']}")
        for title, content in entry["documents"]:
            document = post(
                base_url,
                f"/clients/{client['id']}/documents",
                {"title": title, "content": content},
                api_key,
            )
            print(f"  document {document['id']} {title}")


if __name__ == "__main__":
    main()
