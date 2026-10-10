import csv
import random
import os

# Create folder if it doesn't exist (handled by the script location anyway)
os.makedirs("data", exist_ok=True)

brands = ["Amul", "Britannia", "Lays", "Pepsi", "Nestle", "Local", "Haldirams", "Tropicana"]
cats = ["Dairy", "Snacks", "Beverages", "Bakery", "Staples"]
base_names = ["Milk", "Bread", "Eggs", "Chips", "Cola", "Butter", "Cheese", "Juice", "Water", "Noodles"]

def generate_dataset(filename, platform, num_rows):
    products = []
    for i in range(num_rows):
        name_base = random.choice(base_names)
        mrp = round(random.uniform(20.0, 500.0), 2)
        discount = random.choice([0, 0, 5, 10, 15, 20, 50])
        sell = round(mrp * (1 - discount/100), 2)
        
        products.append({
            "product_id": f"SKU_{random.randint(10000, 99999)}",
            "product_name": f"{random.choice(brands)} {name_base} {random.choice(['500g', '1L', '200g', 'Pack of 2'])}",
            "brand": random.choice(brands),
            "category": random.choice(cats),
            "mrp": mrp,
            "selling_price": sell,
            "discount_pct": discount,
            "availability": random.choice(["In Stock", "In Stock", "In Stock", "Low Stock", "Out of Stock"]),
            "inventory": random.randint(0, 50) if random.random() > 0.1 else 0,
            "rating": round(random.uniform(3.0, 5.0), 1),
            "delivery_eta": random.randint(8, 25),
            "platform": platform,
            "sales_volume": random.randint(100, 1000)
        })
        
    field_names = list(products[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=field_names)
        w.writeheader()
        w.writerows(products)
    print(f"Created {filename} with {num_rows} records.")

def generate_luxury_dataset(filename, platform, num_rows):
    products = []
    luxury_brands = ["Apple", "Samsung", "Sony", "Bose", "Dyson", "LG", "Marshall"]
    luxury_cats = ["Electronics", "Appliances", "Audio", "Smart Home"]
    luxury_bases = ["Phone", "Headphones", "Speaker", "Vacuum", "TV", "Watch", "Tablet"]
    
    for i in range(num_rows):
        name_base = random.choice(luxury_bases)
        mrp = round(random.uniform(5000.0, 150000.0), 2)  # Much higher prices
        discount = random.choice([0, 0, 0, 5, 10])  # Less discounts
        sell = round(mrp * (1 - discount/100), 2)
        
        products.append({
            "product_id": f"LUX_{random.randint(10000, 99999)}",
            "product_name": f"{random.choice(luxury_brands)} {name_base} Pro Max",
            "brand": random.choice(luxury_brands),
            "category": random.choice(luxury_cats),
            "mrp": mrp,
            "selling_price": sell,
            "discount_pct": discount,
            "availability": random.choice(["In Stock", "Low Stock", "Low Stock", "Out of Stock"]),  # More scarcity
            "inventory": random.randint(0, 10) if random.random() > 0.3 else 0, # Lower inventory
            "rating": round(random.uniform(4.0, 5.0), 1), # Generally higher ratings
            "delivery_eta": random.randint(24, 72), # Takes longer to deliver
            "platform": platform,
            "sales_volume": random.randint(5, 50) # Lower volume
        })
        
    field_names = list(products[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=field_names)
        w.writeheader()
        w.writerows(products)
    print(f"Created {filename} with {num_rows} records.")

generate_dataset("dataset1_retail_q1.csv", "PlatformA", 250)
generate_dataset("dataset2_retail_q2.csv", "PlatformB", 300)
generate_luxury_dataset("dataset3_luxury_electronics.csv", "PremiumStore", 150)

