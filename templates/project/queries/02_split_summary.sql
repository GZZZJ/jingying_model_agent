-- 按 final_flag 查看样本量、时间范围和基础标签浓度。
select
  final_flag,
  min({{time_column}}) as min_mdl_dte,
  max({{time_column}}) as max_mdl_dte,
  min({{period_column}}) as min_ds,
  max({{period_column}}) as max_ds,
  count(1) as cnt,
  avg(cast({{target_column}} as double)) as target_rate
from {{sample_table}}
group by final_flag
order by final_flag;
