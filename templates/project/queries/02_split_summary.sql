-- 按 final_flag 查看样本量、时间范围和基础标签浓度。
select
  final_flag,
  min({{time_column}}) as min_mdl_dte,
  max({{time_column}}) as max_mdl_dte,
  min({{period_column}}) as min_ds,
  max({{period_column}}) as max_ds,
  count(1) as cnt,
  avg(cast({{target_column}} as double)) as ftr_30d_ord_rate,
  avg(cast(ftr_30d_ord_amt as double)) as avg_ftr_30d_ord_amt,
  avg(cast(gcard_v6 as double)) as avg_gcard_v6
from {{sample_table}}
group by final_flag
order by final_flag;
