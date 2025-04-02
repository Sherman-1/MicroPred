import polars as pl 


lf = pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/new_data_logits_dataframe.parquet").head(1000)


a = lf.with_columns(
    (1 / (1 + pl.col("logits").neg().exp())).alias("sigmoid")
)

print(a.collect())