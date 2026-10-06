# Decision layer: SHIP / DON'T SHIP / INCONCLUSIVE plus the reasons.
# Refusal rules run first, so a CI on its own never gets turned into a verdict.

import math
from dataclasses import dataclass

SHIP = 'SHIP'
DONT_SHIP = "DON'T SHIP"
INCONCLUSIVE = 'INCONCLUSIVE'


@dataclass
class Decision:
    verdict: str
    reasons: list


def fmt_p(p):
    return 'p < 0.0001' if p < 1e-4 else f'p = {p:.4f}'


def fmt_ci(est):
    return f'[{est.ci_low:+.1%}, {est.ci_high:+.1%}]'


def decide(lift, srm, mde_rel, min_effect=0.05, balance=None, guardrails=None):
    """lift       readout Estimate for the relative lift of the primary metric
    srm        output of trust.srm_test
    mde_rel    relative MDE of the primary metric at the sample size we actually have
    min_effect smallest relative change that matters (in either direction)
    balance    {name: trust.balance_test output} for pre-treatment features, optional
    guardrails {name: lift Estimate} for metrics that must not get significantly worse

    Refusal rules go first and all of them are collected, so you see every problem at once.
    Only if none fire does the CI get turned into SHIP / DON'T SHIP.

    Low power only blocks a result that isn't significant: a null from a test that couldn't
    have seen the effect says nothing. A significant result from an underpowered test still
    gets a verdict, with a warning, since those tend to overstate the effect."""
    refusals = []
    if not all(math.isfinite(v) for v in [lift.value, lift.ci_low, lift.ci_high, mde_rel]):
        refusals.append('the lift is undefined here (no events in control, or too few users)')
    if not srm['passed']:
        refusals.append(
            f"sample ratio mismatch: treatment share {srm['share']:.4f} vs "
            f"{srm['expected_share']:.2f} expected ({fmt_p(srm['p_value'])}), "
            'so the assignment or the logging is broken')
    for name, b in (balance or {}).items():
        if not b['passed']:
            refusals.append(
                f'treatment share differs across {name} bins '
                f"({min(b['shares']):.1%} to {max(b['shares']):.1%}, {fmt_p(b['p_value'])}), "
                'so the arms were different before treatment and the difference in means '
                'is confounded')
    significant = lift.ci_low > 0 or lift.ci_high < 0
    underpowered = mde_rel > min_effect
    if underpowered and not significant:
        refusals.append(
            f'underpowered: this sample can only reliably detect lifts of {mde_rel:.2%} or more, '
            f'bigger than the {min_effect:.2%} that matters, so a null result says little')
    if lift.ci_low < -min_effect and lift.ci_high > min_effect:
        refusals.append(
            f'the CI {fmt_ci(lift)} includes both a meaningful drop and a meaningful gain '
            f'(+-{min_effect:.0%})')
    if refusals:
        return Decision(INCONCLUSIVE, refusals)

    harmed = {name: g for name, g in (guardrails or {}).items() if g.ci_high < 0}
    if harmed:
        return Decision(DONT_SHIP, [
            f'guardrail {name} got significantly worse: {g.value:+.1%} {fmt_ci(g)}'
            for name, g in harmed.items()])

    warning = []
    if underpowered:
        warning = [f'careful: the test was underpowered for a {min_effect:.0%} lift '
                   f'(MDE {mde_rel:.1%}), and significant results from underpowered tests tend '
                   'to overstate the effect']
    if lift.ci_low > 0:
        reasons = [f'primary metric up {lift.value:+.1%}, 95% CI {fmt_ci(lift)}']
        if guardrails:
            reasons.append(f"no guardrail got significantly worse ({', '.join(guardrails)})")
        return Decision(SHIP, reasons + warning)
    if lift.ci_high < 0:
        return Decision(DONT_SHIP,
                        [f'primary metric down {lift.value:+.1%}, 95% CI {fmt_ci(lift)}'] + warning)
    return Decision(DONT_SHIP, [
        f'no significant change ({lift.value:+.1%}, 95% CI {fmt_ci(lift)}). '
        f'the test could detect a {mde_rel:.1%} lift, so an effect that matters would likely '
        'have shown up'])
