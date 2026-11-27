# features

## question

Did the feature step keep every row and the declared feature surface?

## decision — no_action

- metric: row_loss_share = 0.0000 [0.0000, 0.0000] 95% (n=78,938, unit=share)
- rule: max 0.0000
- action (olist): fix the feature transform before training

![figure](features.png)

## evidence

- baseline: row-preserving transform (every input row survives)

### segments

| segment | value | n |
| --- | ---: | ---: |
| train_split | 0.0000 | 78,938 |
| val_split | 0.0000 | 19,735 |

### sensitivity

- features_added_n78938: 6.0000
- features_removed_n78938: 0.0000
- plan_columns_n78938: 18.0000
- first_half_row_loss_n39469: 0.0000

### falsifier

the row-loss interval clears the configured drop ceiling

### provenance

- artifact: feature_pipeline.pkl
- dataset: olist
- plan_columns: items_count, sellers_count, products_count, total_price, total_freight, mean_item_price, mean_product_weight_g, mean_product_photos_qty, mean_product_description_length, total_payment, max_installments, payment_type_count, customer_state_frequency_enc, customer_city_frequency_enc, customer_zip_code_prefix_frequency_enc, dominant_payment_type_frequency_enc, dominant_category_frequency_enc, seller_state_frequency_enc
- columns_added: customer_city_frequency_enc, customer_state_frequency_enc, customer_zip_code_prefix_frequency_enc, dominant_category_frequency_enc, dominant_payment_type_frequency_enc, seller_state_frequency_enc
- columns_removed: none

## tables

- none
