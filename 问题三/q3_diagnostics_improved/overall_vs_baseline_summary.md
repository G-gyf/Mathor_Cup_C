# 第三问改进版与基准版对照

- 样本总数：`278`
- 平均 S6 变化：`0.9994`
- 平均总成本变化：`-86.76` 元
- 平均总负荷变化：`-8.29`

## 样本1/2/3对照
```csv
sample_id,baseline_initial_action,improved_initial_action,baseline_S6,improved_S6,delta_S6,baseline_total_cost,improved_total_cost,delta_total_cost,baseline_total_load,improved_total_load,delta_total_load,baseline_first_level1_month,improved_first_level1_month
1,"I=1, f=10","I=1, f=9",47.04588099999999,48.3192318352413,1.2733508352413097,1050.0,990.0,-60.0,60,55,-5,2,2
2,"I=2, f=10","I=2, f=9",35.168590077952004,36.15185125881361,0.9832611808616036,1380.0,1280.0,-100.0,120,110,-10,1,1
3,"I=3, f=5","I=3, f=8",30.969193704154,30.979776930732754,0.010583226578752658,1990.0,1862.0,-128.0,165,153,-12,1,1
```