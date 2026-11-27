# Olist EDA and forecasting — distilled reference

Sources (read, not copied):

- `marcopeix/youtube_tutorials` — `YT_03_forecasting_stats.ipynb` (Nixtla `statsforecast`).
- Kaggle `kabure/simple-eda-sales-and-customer-patterns`.
- Kaggle `jsaguiar/e-commerce-exploratory-analysis`.
- Dataset: `olistbr/brazilian-ecommerce` (9 CSVs, 2016-09 → 2018-10).

This is the checklist and the expected shape of the answers, not the notebooks' code.

## 1. Data model (join keys)

```
orders (1) ──< order_items >── products, sellers
   │ 1
   ├──< payments        (payment_sequential; an order can split payment)
   ├──< reviews         (one review per order in practice)
   └── customers (customer_id unique per order; customer_unique_id = person)
geolocation: zip-prefix → lat/lng/city/state (1M rows, many duplicates per zip)
product_category_name_translation: PT → EN
```

Rules that prevent most EDA errors:

- Joining `orders` to items/payments/reviews creates **multiple rows per order**. Decide the grain first: order, order-item, payment, or review. Never count orders on a joined frame without `nunique(order_id)`.
- `customer_id` is per order; repeat customers are counted on `customer_unique_id`.
- Delivery timestamps are nullable (`order_delivered_customer_date` missing ~3%); filter to `delivered` for delivery-time questions and say what was dropped.
- Prices/freight are BRL. Review scores are 1–5 and imbalanced.

## 2. EDA checklist, in order

**Volume and money**

- Orders, items, payments, reviews counts; unique customers; date range; monthly volume.
- Price distribution: mean ≈ 120.65, std ≈ 184.11 (right-skewed, log is the honest view); freight share of total.
- Payments: credit card ≈ 73.5–75%, boleto ≈ 19.5–20%; installments only on credit card, up to ~10; average order value rises with installments up to 10. A single-installment order has a different distribution — segment it.
- Order items: ≈ 87% of orders have one item; long tail. Split multi-item orders before per-item conclusions.

**Time**

- Daily/weekly revenue and order count. Expected shape: sparse 2016, growth through 2017, **Black Friday spike (24 Nov)**, sharp drop after ~20 Dec (holidays), 2018 growth then the dataset ends mid-Oct.
- Seasonality: weekday vs weekend, month-of-year, holiday windows. Do not fit a seasonal model on 2016 alone (almost no data).

**Geography**

- Customer states: SP ≈ 42%, RJ ≈ 13%, MG ≈ 11.6%. Seller states: SP ≈ 70%.
- Freight matrix seller-state × customer-region: SP sellers have the lowest mean freight to every region; long-distance lanes (north/northeast) are the expensive ones.
- Geolocation: zip-level only; deduplicate before mapping.

**Products**

- 71 categories (PT + EN translation). Category concentration is heavy; group the tail (<1000 orders) into "other".
- Photos, description length, weight/dimensions: distributions and relation to price/freight and review score.
- `product_name_lenght` / `product_description_lenght` are misspelled in the source — keep the source spelling in the contract.

**Reviews**

- Score distribution: 5 ≈ 57k, 4 ≈ 19k, 3 ≈ 8k, 2 ≈ 3k, 1 ≈ 11k. The 1-star mass is the interesting one.
- Bad scores correlate with **longer delivery time** and slightly more expensive/larger orders. Delivery time has extreme outliers; cap the view (e.g. 60 days) and report the cap.
- Review comments are mostly null; text is not the first target.

**Delivery**

- Compute `delivery_days = delivered_customer - purchase`, `carrier_days`, `late = delivered > estimated`.
- Lead time and late rate by state/lane, category, seller, month. Late delivery is the strongest review predictor to test.

**Customers**

- Repeat purchase via `customer_unique_id`; most customers buy once. Order frequency, tenure, state.

## 3. Forecasting workflow (from YT_03, adapted)

Grain: one series (daily orders or daily revenue), horizon 7 (or 28). Split chronologically, never randomly.

1. **Baseline first** — seasonal naive (same weekday last week) and historic mean. Report MAE/RMSE/MAPE on the holdout. No model ships without beating it.
2. **Statistical model** — AutoARIMA / SARIMA with weekly seasonality (7). One model per series if multiple (e.g. by state/category).
3. **Cross-validation** — multiple rolling/expanding windows, not a single split. Report mean and spread across folds.
4. **Exogenous features** — only future-known columns: calendar (weekday, month, holiday, Black Friday), price/freight if a price plan exists. Do not feed the target or contemporaneous outcomes.
5. **Prediction intervals** — probabilistic forecast; check empirical coverage on the CV folds, not just the point error.
6. **Metrics** — MAE/RMSE for scale, MAPE only when the series has no zeros. Compare per fold and aggregate.

Pitfalls the notebook makes visible:

- A single holdout is one sample; CV is the evidence.
- Seasonal naive is strong on weekly retail data — AutoARIMA must earn its place.
- Exogenous features must be available at forecast time; otherwise they are leakage.
- Intervals that are not checked for coverage are decoration.

## 4. What to decide before building

- Track: **daily order count** (stable, seasonality) vs **daily revenue** (money, outliers).
- ML track (if any): `review_score` classification (imbalanced) or `delivery_days`/`late` regression.
- Grain for joins: order vs order-item vs review.
- Horizon and frequency; which exogenous columns are genuinely future-known.
- What to do with 2016 (exclude from training, keep in the plot) and with the final partial month.

## 5. Expected findings to verify, not assume

The notebooks report: SP concentration on both sides, credit-card dominance, single-item orders, Black Friday spike and holiday dip, delivery time tracking review score, and freight cost driven by lane. Treat these as hypotheses to re-measure on the loaded data, with n and an interval, before they enter a decision record.
