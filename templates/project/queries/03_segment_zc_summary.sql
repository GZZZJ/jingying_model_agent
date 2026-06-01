-- 按客群和风险等级查看样本分布、标签浓度和历史分数。
select
  final_flag,
  blue_customer_flag,
  zc_level,
  count(1) as cnt,
  avg(cast({{target_column}} as double)) as ftr_30d_ord_rate,
  avg(cast(gcard_v6 as double)) as avg_gcard_v6
from {{sample_table}}
group by final_flag, blue_customer_flag, zc_level
order by final_flag, blue_customer_flag, zc_level;
