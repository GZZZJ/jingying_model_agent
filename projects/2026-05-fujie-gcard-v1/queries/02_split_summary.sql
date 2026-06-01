-- 按 final_flag 查看样本量、时间范围和基础标签浓度。
select
  final_flag,
  min(mdl_dte) as min_mdl_dte,
  max(mdl_dte) as max_mdl_dte,
  min(ds) as min_ds,
  max(ds) as max_ds,
  count(1) as cnt,
  avg(cast(ftr_30d_ord_flag as double)) as ftr_30d_ord_rate,
  avg(cast(ftr_30d_ord_amt as double)) as avg_ftr_30d_ord_amt,
  avg(cast(gcard_v6 as double)) as avg_gcard_v6
from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1
group by final_flag
order by final_flag;
