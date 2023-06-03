import pandas as pd

mql = pd.read_csv("data/olist_marketing_qualified_leads_dataset.csv")
cd = pd.read_csv("data/olist_closed_deals_dataset.csv")

mf = mql.merge(cd, on="mql_id", how="left")

sellers = pd.read_csv("data/olist_sellers_dataset.csv")

mf_sellers = mf.merge(sellers, on="seller_id", how="left")

items = pd.read_csv("data/olist_order_items_dataset.csv")

mf_items = mf.merge(items, on="seller_id", how="left")

mf_sellers.to_csv("data/sellers_dataset.csv", index=False, header=True)
mf_items.to_csv("data/items_dataset.csv", index=False, header=True)
