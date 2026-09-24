-- Metric definitions, one row per arm. Everything downstream reads these views instead of
-- the raw table, the same way an experimentation platform has one place where metrics are
-- defined. Sums are kept (not just rates) so python can get variances without pulling 14M rows.
-- Analysis is by assignment (treatment), never by exposure, see exposed_vs_control below.

create or replace view arm_sizes as
select
    treatment,
    count(*) as n
from criteo
group by treatment;

create or replace view visit_rate as
select
    treatment,
    count(*) as n,
    sum(visit)::bigint as visits,
    avg(visit) as visit_rate
from criteo
group by treatment;

create or replace view conversion_rate as
select
    treatment,
    count(*) as n,
    sum(conversion)::bigint as conversions,
    avg(conversion) as conversion_rate
from criteo
group by treatment;

-- Ratio metric. The unit of randomization is the user, not the visit, so the variance
-- has to come from per-user sums (delta method in readout.py). Visits and conversions are
-- 0/1 so their squares are just the counts, but conv * visit is kept in case that changes.
create or replace view conversions_per_visit as
select
    treatment,
    count(*) as n,
    sum(conversion)::bigint as conversions,
    sum(visit)::bigint as visits,
    sum(conversion * visit)::bigint as conv_visits,
    sum(conversion) / sum(visit) as conversions_per_visit
from criteo
group by treatment;

-- NOT for decisions. Only treated users can be exposed and exposure isn't random (it depends
-- on whether the user showed up in an auction), so comparing exposed users to all of control
-- mixes the ad effect with who gets exposed. Kept to show how big that bias is.
create or replace view exposed_vs_control as
select
    case when treatment = 0 then 'control'
         when exposure = 1 then 'treated, exposed'
         else 'treated, not exposed' end as grp,
    count(*) as n,
    avg(visit) as visit_rate,
    avg(conversion) as conversion_rate
from criteo
group by grp;
