-- 按客群和风险等级查看样本分布、标签浓度。
select
  final_flag,
  blue_customer_flag,
  zc_level,
  count(1) as cnt,
  avg(cast({{target_column}} as double)) as target_rate
from {{sample_table}}
group by final_flag, blue_customer_flag, zc_level
order by final_flag, blue_customer_flag, zc_level;
