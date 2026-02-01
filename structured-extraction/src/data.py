"""Data preparation utilities for receipt extraction."""

import json
import random
from pathlib import Path

import click


# Sample receipt templates for generating test data
RECEIPT_TEMPLATES = [
    {
        "text": """STARBUCKS COFFEE
123 Main Street
Seattle, WA 98101

Date: 2024-01-15
Time: 08:32

Grande Latte           $5.75
Blueberry Muffin       $3.25
----------------------------
Subtotal:              $9.00
Tax (10%):             $0.90
----------------------------
TOTAL:                $9.90

Paid by: VISA ****1234

Thank you for visiting!""",
        "ground_truth": {
            "vendor_name": "Starbucks Coffee",
            "date": "2024-01-15",
            "subtotal": 9.00,
            "tax": 0.90,
            "total": 9.90,
        },
    },
    {
        "text": """WALMART SUPERCENTER
456 Commerce Blvd
Austin, TX 78701
Tel: (512) 555-0100

01/20/2024  14:45

BANANAS 2.5LB      $1.47
MILK 1GAL          $3.98
BREAD WHITE        $2.48
EGGS LARGE DZ      $4.29
CHICKEN BREAST     $8.99

SUBTOTAL          $21.21
TAX                $1.75
TOTAL             $22.96

CASH TEND         $25.00
CHANGE DUE         $2.04

ITEMS SOLD: 5
TC# 4582-9371-2845-1234""",
        "ground_truth": {
            "vendor_name": "Walmart Supercenter",
            "date": "2024-01-20",
            "subtotal": 21.21,
            "tax": 1.75,
            "total": 22.96,
        },
    },
    {
        "text": """CVS pharmacy
789 Health Ave
Boston, MA 02101

Store #4521
02-08-2024 16:22

TYLENOL EXTRA STR     $12.99
VITAMIN C 1000MG       $8.49
BANDAGES ASST          $5.99
------------------------
SUBTOTAL              $27.47
SALES TAX              $1.65
------------------------
TOTAL                 $29.12

DEBIT CARD           $29.12

Thank you for shopping at CVS!
Save more with ExtraCare!""",
        "ground_truth": {
            "vendor_name": "CVS Pharmacy",
            "date": "2024-02-08",
            "subtotal": 27.47,
            "tax": 1.65,
            "total": 29.12,
        },
    },
    {
        "text": """TARGET
1000 Retail Way
Chicago, IL 60601

03/15/2024
11:05 AM

T-SHIRT MENS BLK    $19.99
SOCKS 6PK            $9.99
TOWELS BATH 2PK     $24.99
SOAP BAR 4PK         $6.49
========================
Subtotal            $61.46
Sales Tax (8.5%)     $5.22
========================
Total               $66.68

Visa Credit
Card#: XXXX-XXXX-XXXX-5678

REDcard save 5% on every purchase!""",
        "ground_truth": {
            "vendor_name": "Target",
            "date": "2024-03-15",
            "subtotal": 61.46,
            "tax": 5.22,
            "total": 66.68,
        },
    },
    {
        "text": """WHOLE FOODS MARKET
Organic & Natural
2500 Green Lane, SF CA 94102

Date: 2024-03-22 Time: 18:30

ORGANIC APPLES        $6.99
ALMOND MILK           $4.49
QUINOA 1LB            $7.99
AVOCADOS (3)          $5.97
OLIVE OIL EVOO       $12.99
SALMON FILLET        $18.99
-----------------------------
SUBTOTAL             $57.42
CA TAX                $4.88
-----------------------------
TOTAL                $62.30

Amazon Prime Member Savings: $4.20

Paid: AMEX ****9012""",
        "ground_truth": {
            "vendor_name": "Whole Foods Market",
            "date": "2024-03-22",
            "subtotal": 57.42,
            "tax": 4.88,
            "total": 62.30,
        },
    },
    {
        "text": """McDONALD'S
I'm Lovin' It
555 Fast Food Drive
Denver, CO 80202

04/10/24 12:15

BIG MAC MEAL         $9.99
  - Medium Fries
  - Medium Coke
MCNUGGETS 10PC       $5.49
APPLE PIE            $1.89
--------------------------
Subtotal:           $17.37
Tax:                 $1.39
--------------------------
TOTAL:              $18.76

Debit Mastercard    $18.76

Order #: 247
Thank you! Come again!""",
        "ground_truth": {
            "vendor_name": "McDonald's",
            "date": "2024-04-10",
            "subtotal": 17.37,
            "tax": 1.39,
            "total": 18.76,
        },
    },
    {
        "text": """HOME DEPOT
You Can Do It. We Can Help.
3000 Builder Blvd
Phoenix, AZ 85001

05/05/2024  09:15:33

2x4x8 LUMBER (10)      $42.50
SCREWS 1LB BOX          $8.97
WOOD STAIN QT          $14.98
PAINT BRUSH 3PK         $9.99
SANDPAPER 5PK           $6.49
================================
SUBTOTAL               $82.93
AZ STATE TAX            $7.05
================================
TOTAL                  $89.98

PRO XTRA MEMBER
Points Earned: 90

VISA DEBIT            $89.98
Approval: 847291""",
        "ground_truth": {
            "vendor_name": "Home Depot",
            "date": "2024-05-05",
            "subtotal": 82.93,
            "tax": 7.05,
            "total": 89.98,
        },
    },
    {
        "text": """COSTCO WHOLESALE
Members Only Warehouse
8000 Bulk Lane
Seattle WA 98108

Member: 123456789012
06/18/2024 14:22

KIRKLAND WATER 40PK  $4.99
ROTISSERIE CHICKEN   $4.99
PAPER TOWELS 12PK   $18.99
LAUNDRY DET 150OZ   $16.99
CHEESE 2LB BLOCK     $8.49
OLIVE OIL 2L        $17.99
--------------------------
SUBTOTAL            $72.44
WA TAX               $7.24
--------------------------
TOTAL               $79.68

COSTCO VISA         $79.68

2% CASHBACK EARNED: $1.59
THANK YOU FOR SHOPPING!""",
        "ground_truth": {
            "vendor_name": "Costco Wholesale",
            "date": "2024-06-18",
            "subtotal": 72.44,
            "tax": 7.24,
            "total": 79.68,
        },
    },
    {
        "text": """BEST BUY
Expert Service. Unbeatable Price.
Store #1542
7500 Electronics Pkwy
San Jose, CA 95110

Date: 07/22/2024
Time: 15:45

APPLE AIRPODS PRO 2   $249.00
USB-C CABLE 6FT         $19.99
SCREEN PROTECTOR        $29.99
----------------------------
Subtotal              $298.98
CA Sales Tax           $26.91
----------------------------
TOTAL                 $325.89

BEST BUY CREDIT      $325.89

My Best Buy Points: 650
Totaltech Member Benefits Applied

Trans#: 5891-7234-9012""",
        "ground_truth": {
            "vendor_name": "Best Buy",
            "date": "2024-07-22",
            "subtotal": 298.98,
            "tax": 26.91,
            "total": 325.89,
        },
    },
    {
        "text": """TRADER JOE'S
Your Neighborhood Grocery
500 Nautical Way
Portland OR 97201

08/30/24  10:08

TJ ORANGE JUICE      $3.99
EVERYTHING BAGELS    $3.49
GREEK YOGURT         $5.99
FROZEN PIZZA         $4.99
DARK CHOC ALMONDS    $4.49
BANANA .42LB          $.17
=========================
SUBTOTAL            $23.12
OR TAX               $0.00
=========================
TOTAL               $23.12

CASH                $25.00
CHANGE               $1.88

Crew Member: Sarah
Have a great day!""",
        "ground_truth": {
            "vendor_name": "Trader Joe's",
            "date": "2024-08-30",
            "subtotal": 23.12,
            "tax": 0.00,
            "total": 23.12,
        },
    },
]


def add_noise(text: str, noise_level: float = 0.1) -> str:
    """Add OCR-like noise to receipt text."""
    if noise_level <= 0:
        return text

    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_level:
            noise_type = random.choice(["delete", "replace", "insert"])
            if noise_type == "delete" and chars[i] not in "\n ":
                chars[i] = ""
            elif noise_type == "replace":
                if chars[i].isdigit():
                    chars[i] = random.choice("0123456789")
                elif chars[i].isalpha():
                    similar = {
                        "O": "0", "0": "O", "l": "1", "1": "l",
                        "S": "5", "5": "S", "B": "8", "8": "B",
                    }
                    chars[i] = similar.get(chars[i], chars[i])
            elif noise_type == "insert":
                chars[i] = chars[i] + random.choice(" .")
    return "".join(chars)


@click.command()
@click.option("--output", "-o", default="data/receipts.jsonl", help="Output file path")
@click.option("--count", "-n", default=50, help="Number of receipts to generate")
@click.option("--noise", default=0.02, help="OCR noise level (0-1)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def generate_sample_data(output: str, count: int, noise: float, seed: int):
    """Generate sample receipt data for testing."""
    random.seed(seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    receipts = []
    for i in range(count):
        template = random.choice(RECEIPT_TEMPLATES)
        text = add_noise(template["text"], noise)

        receipts.append({
            "id": f"receipt_{i:04d}",
            "text": text,
            "ground_truth": template["ground_truth"],
        })

    with open(output_path, "w") as f:
        for receipt in receipts:
            f.write(json.dumps(receipt) + "\n")

    click.echo(f"Generated {count} receipts to {output_path}")


if __name__ == "__main__":
    generate_sample_data()
