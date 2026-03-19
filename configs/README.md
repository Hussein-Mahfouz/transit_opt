# Config notes

- `config_template.yaml` is the starting point for any one wanting to run the optimisaion pipeline.

I use multiple configs to run different scenarios. I have created a naming convention for these configs.


### Components

| Element | Description | Example Values | Example Meaning |
|----------|--------------|----------------|-----------------|
| **objective** | The optimisation objective being evaluated | `wt` = WaitingTimeObjective<br>`sc` = StopCoverageObjective | `wt` → objective minimises user waiting time |
| **time_aggregation** | How values are aggregated across time intervals | `av` = average<br>`pk` = peak<br>`sm` = sum<br>`int` = intervals | `av` → uses average values across intervals |
| **metric** | The performance measure used in the objective | `tot` = total<br>`var` = variance | `tot` → total waiting time or total vehicles |

### Examples

| Config file | Expanded meaning |
|--------------|------------------|
| `wt_av_tot.yaml` | WaitingTimeObjective, time aggregation = *average*, metric = *total* |
| `wt_pk_var.yaml` | WaitingTimeObjective, time aggregation = *peak*, metric = *variance* |
| `sc_av_var.yaml` | StopCoverageObjective, time aggregation = *average*, metric = *variance* |
| `sc_int_var.yaml` | StopCoverageObjective, time aggregation = *intervals*, metric = *variance* |
