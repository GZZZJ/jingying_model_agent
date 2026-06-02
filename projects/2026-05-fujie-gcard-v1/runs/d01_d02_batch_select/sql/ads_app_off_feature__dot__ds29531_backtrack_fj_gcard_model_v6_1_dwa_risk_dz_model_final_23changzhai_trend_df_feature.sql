select *
from ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_dwa_risk_dz_model_final_23changzhai_trend_df_feature
where ds = '20250630'
and rand_flag0 < 0.1
and final_flag in ('DEV', 'OOT')
union all
select *
from ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_dwa_risk_dz_model_final_23changzhai_trend_df_feature
where ds = '20260131'
and rand_flag0 < 0.1
and final_flag in ('DEV', 'OOT')
