import csv
import random
import os

os.makedirs("data", exist_ok=True)

# Define rules so that Machine Learning can actually find patterns!
# Brand is strictly tied to Category, and price ranges are specific.
brand_rules = {
    "Amul": {"cat": "Dairy", "bases": ["Milk", "Butter", "Cheese"], "price_range": (30, 150)},
    "Nestle": {"cat": "Dairy", "bases": ["Milk", "Yogurt"], "price_range": (40, 180)},
    "Lays": {"cat": "Snacks", "bases": ["Chips", "Nachos"], "price_range": (20, 80)},
    "Haldirams": {"cat": "Snacks", "bases": ["Bhujia", "Mixture"], "price_range": (50, 120)},
    "Pepsi": {"cat": "Beverages", "bases": ["Cola", "Soda"], "price_range": (40, 90)},
    "Tropicana": {"cat": "Beverages", "bases": ["Juice", "Smoothie"], "price_range": (100, 250)},
    "Britannia": {"cat": "Bakery", "bases": ["Bread", "Cookies"], "price_range": (30, 100)},
}

def generate_smart_dataset(filename, num_rows):
    products = []
    brands = list(brand_rules.keys())
    
    for i in range(num_rows):
        # 1. Pick a brand
        brand = random.choice(brands)
        rules = brand_rules[brand]
        
        # 2. Features strictly follow the brand's rules (Strong Correlation)
        category = rules["cat"]
        name_base = random.choice(rules["bases"])
        mrp = round(random.uniform(rules["price_range"][0], rules["price_range"][1]), 2)
        
        discount = random.choice([0, 5, 10, 15])
        sell = round(mrp * (1 - discount/100), 2)
        
        products.append({
            "product_id": f"SMART_{random.randint(10000, 99999)}",
            "product_name": f"{brand} {name_base} {random.choice(['500g', '1L', 'Pack'])}",
            "brand": brand,
            "category": category,
            "mrp": mrp,
            "selling_price": sell,
            "discount_pct": discount,
            "availability": random.choice(["In Stock", "In Stock", "Low Stock"]),
            "inventory": random.randint(10, 100),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "delivery_eta": random.randint(10, 30),
            "platform": "SmartMart",
            "sales_volume": int(mrp * 2) # Correlate sales volume with price loosely
        })
        
    field_names = list(products[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=field_names)
        w.writeheader()
        w.writerows(products)
    print(f"Created {filename} with {num_rows} records.")

generate_smart_dataset("dataset4_high_accuracy.csv", 400)
