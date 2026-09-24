-- Segment cuts by decile of f0 and f2 (features are anonymized, these two have the most
-- distinct values and the uplift clearly varies across them).
-- Cutpoints are by value, so big ties merge deciles: f0 ends up with 9 bins and f2 with 6.
-- bin = how many of the 9 decile cutpoints the value is above (0 to 9).

create or replace table cutpoints as
select
    quantile_cont(f0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) as f0_cuts,
    quantile_cont(f2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) as f2_cuts
from criteo;

create or replace view criteo_segments as
select
    criteo.*,
    len(list_filter(f0_cuts, c -> c < f0)) as f0_bin,
    len(list_filter(f2_cuts, c -> c < f2)) as f2_bin
from criteo, cutpoints;

-- f0 bin x f2 bin x arm is the finest level. python and the app roll this up to whatever
-- segment they need, so the 14M rows only get scanned once.
create or replace view segment_stats as
select
    f0_bin,
    f2_bin,
    treatment,
    count(*) as n,
    sum(visit)::bigint as visits,
    sum(conversion)::bigint as conversions,
    sum(conversion * visit)::bigint as conv_visits
from criteo_segments
group by f0_bin, f2_bin, treatment;
