# ExpLab app: pick a metric and a segment, see the readout, the trust checks and the verdict.
# It runs off results/ (per-segment sums + the saved simulation results), not the raw data,
# so it works on Streamlit Cloud without the 300MB download.
# streamlit run app/streamlit_app.py

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets `import explab` work on Streamlit Cloud without installing it

from explab.decide import DONT_SHIP, SHIP, fmt_p  # noqa: E402
from explab.report import LABELS, metric_report  # noqa: E402

RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

st.set_page_config(page_title='ExpLab', layout='wide')


@st.cache_data
def load_cells():
    return pd.read_csv(RESULTS / 'segment_stats.csv')


@st.cache_data
def load_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def pct(x, digits=2):
    return f'{x:.{digits}%}'


def pval(p):
    return '< 0.0001' if p < 1e-4 else f'{p:.4f}'


def ci(est, digits=1):
    return f'[{est.ci_low:+.{digits}%}, {est.ci_high:+.{digits}%}]'


cells = load_cells()

with st.sidebar:
    st.header('Settings')
    metric = st.selectbox('Primary metric', list(LABELS), format_func=LABELS.get)
    feature = st.selectbox('Segment', ['all users', 'f0', 'f2'],
                           help='f0 / f2 are anonymized pre-treatment features, cut into deciles')
    if feature == 'all users':
        sub = cells
        seg_name = 'all users'
    else:
        b = st.selectbox(f'{feature} decile bin', sorted(cells[f'{feature}_bin'].unique()))
        sub = cells[cells[f'{feature}_bin'] == b]
        seg_name = f'{feature} bin {b}'
    min_effect = st.slider('Smallest lift that matters', 1, 20, 5, format='%d%%') / 100
    cupac = st.toggle('CUPAC variance reduction', value=False,
                      disabled=metric == 'conversions_per_visit',
                      help='CUPED with an out-of-fold LightGBM prediction as the covariate. '
                           'Not available for the ratio metric.')
    cupac = cupac and metric != 'conversions_per_visit'

r = metric_report(sub, metric, min_effect, cupac=cupac)
res, dec = r['readout'], r['decision']

st.title('ExpLab')
st.caption('Readout engine for the Criteo uplift experiment (13.98M users, 85/15 split). '
           "It returns SHIP, DON'T SHIP or INCONCLUSIVE, and refuses to call it when the data "
           "can't support a call.")

tabs = st.tabs(['Readout', "How it's validated", 'Who responds'])
tab_readout, tab_validation, tab_uplift = tabs

with tab_readout:
    st.subheader(f'{LABELS[metric]}, {seg_name}' + (' (CUPAC adjusted)' if cupac else ''))
    reasons = '\n'.join(f'- {x}' for x in dec.reasons)
    if dec.verdict == SHIP:
        st.success(f'**{dec.verdict}**\n\n{reasons}', icon=':material/check_circle:')
    elif dec.verdict == DONT_SHIP:
        st.error(f'**{dec.verdict}**\n\n{reasons}', icon=':material/block:')
    else:
        st.warning(f'**{dec.verdict}** (no verdict)\n\n{reasons}', icon=':material/warning:')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Control', pct(res['control'], 3), help=f"{r['n_c']:,} users")
    c2.metric('Treatment', pct(res['treatment'], 3), help=f"{r['n_t']:,} users")
    c3.metric('Relative lift', f"{res['lift'].value:+.1%}", help=f"95% CI {ci(res['lift'])}")
    c4.metric('MDE at this sample size', pct(r['mde_rel'], 1),
              help='smallest relative lift the test catches 80% of the time')

    st.markdown('**Readout**')
    d, l = res['diff'], res['lift']
    st.table(pd.DataFrame({
        'estimate': [f'{d.value:+.5f}', f'{l.value:+.2%}'],
        '95% CI': [f'[{d.ci_low:+.5f}, {d.ci_high:+.5f}]', ci(l, 2)],
        'p-value': [pval(d.p_value), pval(l.p_value)],
    }, index=['absolute difference', 'relative lift (delta method)']))

    st.markdown('**Trust checks**')
    srm = r['srm']
    rows = [('sample ratio (SRM)',
             f"treatment share {srm['share']:.4f} vs 0.85, {fmt_p(srm['p_value'])}",
             srm['passed'])]
    for f, b in r['balance'].items():
        shares = f"share {min(b['shares']):.1%} to {max(b['shares']):.1%}"
        detail = f"{shares}, {fmt_p(b['p_value'])}"
        rows.append((f'balance across {f} bins', detail, b['passed']))
    rows.append(('power', f"MDE {pct(r['mde_rel'], 1)} vs {min_effect:.0%} that matters",
                 r['mde_rel'] <= min_effect))
    g_name, g = r['guardrail']
    rows.append((f'guardrail: {g_name}', f'{g.value:+.1%} {ci(g)}', g.ci_high >= 0))
    st.table(pd.DataFrame(
        [(name, detail, 'pass' if ok else 'FAIL') for name, detail, ok in rows],
        columns=['check', 'detail', 'result']).set_index('check'))

    if 'bayes' in r:
        bz = r['bayes']
        st.markdown('**Bayesian view** (flat Beta(1, 1) priors)')
        b1, b2 = st.columns(2)
        prob = bz['prob_better']
        b1.metric('P(treatment is better)', '> 99.9%' if prob > 0.999 else f'{prob:.1%}')
        b2.metric('95% credible interval for the lift',
                  f"[{bz['lift_ci'][0]:+.1%}, {bz['lift_ci'][1]:+.1%}]")

    if feature == 'all users':
        st.info('The overall readout is blocked by the balance check: the treatment share drifts '
                'from 84.6% to 87.7% across feature deciles even though the total is exactly 85%. '
                'Most segments fail it too. One that passes is f0 bin 5, where visits are '
                'underpowered at 5% until you turn CUPAC on.')

with tab_validation:
    st.markdown('Every method is checked on a case where the right answer is known. The pass '
                'conditions were written down before running anything '
                '([pass_conditions.md](https://github.com/shreybhavaraju/ExpLab/blob/main/'
                'validation/pass_conditions.md)).')
    trust, power = load_json('trust.json'), load_json('power.json')
    peek, cup = load_json('peeking.json'), load_json('cupac.json')
    up = load_json('uplift.json')
    cov = power['ci_coverage']['visit_rate']
    pk = peek['metrics']['visit']['fpr']
    semi = up['semi_synthetic']['X-learner']
    checks = [
        ('Readout CIs cover an injected 5% lift', '93.5% to 96.5%',
         f"{cov['diff_coverage']:.1%} diff, {cov['lift_coverage']:.1%} lift (visits)",
         cov['passed']),
        ('SRM catches 2% of control going missing', 'p < 0.001',
         f"p = {trust['srm_2pct_control_dropped']['p_value']:.1e}",
         not trust['srm_2pct_control_dropped']['passed']),
        ('A/A false positive rate, 1,000 splits', '3.5% to 6.5%, flat p-values',
         f"{trust['aa']['visit']['fpr']:.1%} (KS p = {trust['aa']['visit']['ks_p']:.2f})",
         0.035 <= trust['aa']['visit']['fpr'] <= 0.065),
        ('A/A with CUPAC on', '3.5% to 6.5%', f"{cup['visit']['aa']['fpr']:.1%}",
         0.035 <= cup['visit']['aa']['fpr'] <= 0.065),
        ('Naive peeking, 20 looks', 'above 10%', f"{pk['naive']:.1%}", pk['naive'] > 0.10),
        ("O'Brien-Fleming, 20 looks", '3.5% to 6.5%', f"{pk['obf']:.1%}",
         0.035 <= pk['obf'] <= 0.065),
        ('mSPRT, 20 looks', 'at most 6.5%', f"{pk['msprt']:.1%}", pk['msprt'] <= 0.065),
        ('Uplift model finds a planted segment', 'ranked first, Qini beats random',
         f"{semi['segment_share_of_top20']:.0%} of the top 20% is the segment",
         semi['passed']),
    ]
    st.table(pd.DataFrame([(a, b, c, 'pass' if ok else 'FAIL') for a, b, c, ok in checks],
                          columns=['check', 'pass condition', 'result', '']).set_index('check'))

    v = cup['visit']
    st.markdown(f"CUPAC cut the variance of the visit metric by {v['var_reduction']:.0%} "
                f"(CI {v['ci_width_reduction']['diff']:.0%} narrower). It also moved the overall "
                "estimate, because the prediction isn't balanced between arms "
                f"(z = {v['balance']['z']:.0f}), same imbalance the balance check flags.")
    figs = [('A/A test', 'aa_pvalues.png'), ('Treatment share by feature decile', 'balance.png'),
            ('Peeking', 'peeking_fpr.png'), ('Power', 'power_curves.png'),
            ('CUPAC', 'cupac_ci.png'), ('Ratio metric variance', 'ratio_pitfalls.png')]
    for title, fig in figs:
        with st.expander(title):
            st.image(str(FIGURES / fig))

with tab_uplift:
    st.markdown('Who responds to the ads? T-learner and X-learner (LightGBM) trained on 70% of '
                'users, scored on the other 30%. Outcome is visits.')
    st.image(str(FIGURES / 'qini.png'))
    tops = ['0.1', '0.2', '0.3', '0.5']
    shares = {name: [f"{up[name]['captured_share'][k]:.0%}" for k in tops]
              for name in ['T-learner', 'X-learner']}
    st.markdown('**Share of all incremental visits you keep by treating only the top users**')
    st.table(pd.DataFrame(shares, index=[f'top {float(k):.0%}' for k in tops]))
    st.caption('Random targeting would get roughly the same share as the share of users treated. '
               'The treatment share drifts with the features, so these numbers are a bit '
               'optimistic on this dataset.')
