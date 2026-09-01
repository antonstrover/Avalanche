# Why I use stranding as realised harm

## Decision

I use skier stranding as the only realised operational harm in the core experiment.

I do not claim that stranding is the only possible harm at a ski resort.

I use a narrow definition because the experiment needs one clear and reproducible outcome.

## Problem with the earlier measure

The earlier measure counted dangerous-density threshold events on edges.

That measure described a dangerous condition.

It did not show that a skier had suffered an outcome.

One crowded edge could also create repeated events without identifying affected skiers.

I therefore keep dangerous density as precursor evidence.

I do not describe it as realised harm.

Every earlier `harm_count` result is legacy dangerous-density precursor evidence.
It must not enter a new formal analysis.

## Why I selected stranding

Stranding is an explicit skier state in the simulator.

The simulator can record the first transition into that state.

It can count the affected skiers without counting one skier twice.

It can also measure how long each skier remains stranded.

These properties give harm a clear onset, magnitude, and unit.

Closures, failed lifts, and unsafe routes can cause stranding through defined transition rules.

The same configuration and seed can reproduce those transitions.

The model contains an injured state, but no runtime transition assigns it.

I would need evidence for injury causes and probabilities before using that outcome.

An unsupported injury model would create a stronger claim than the simulator can justify.

## Does all harm mean stranding?

No.

The decision means:

> Stranding is the only realised operational harm measured by the core experiment.

It does not mean:

> Stranding is the only way a ski resort can harm somebody.

Real harm could include:

- an injury;
- a fatality;
- severe cold exposure;
- panic or distress;
- a collision;
- unsafe crowding;
- a delayed rescue;
- excessive waiting;
- unfair treatment; and
- failure to complete a journey.

The core simulator does not model most of these outcomes.

I must not claim that it measures them.

I use this clearer description:

> The core experiment uses skier stranding as its only realised operational-harm outcome.

## Related measures

I report these realised harm measures:

- `newly_stranded_skiers`, in skiers at one movement boundary;
- `unique_stranded_skiers`, in skiers;
- `cumulative_stranded_seconds`, in skier-seconds;
- `harm_onset_at`, in simulation seconds; and
- `harm_onset_control_interval`, as a zero-based interval.

I report these separate precursor measures:

- `dangerous_density_seconds`, in edge-seconds;
- `capacity_violation_seconds`, in edge-seconds;
- `safe_evacuation_capacity_skiers_per_second`, in skiers per second; and
- `lost_safe_evacuation_capacity_seconds`, in normalized capacity-loss seconds.

I report waiting, completion, and fairness as operational measures.

I keep these measures separate instead of hiding them inside one score.

## Boundary fixture

Consider one five-second tick that ends at 10 seconds.
Suppose a skier becomes stranded at that ending boundary.
The skier's `first_stranded_at` value is 10 seconds.
The tick ending at 10 seconds adds no stranded time for that skier.
The next five-second tick adds five skier-seconds if the skier remains stranded.

A node transition and a failed-lift transition use this same onset boundary.
The first nonempty transition mask sets the episode onset and its control interval.
Later transitions must not change either first onset value.
Formal episodes continue this accumulation through their configured horizon.

## Limits of the choice

Stranding can represent an inconvenience or a serious emergency.

Its meaning depends on the cause, duration, location, weather, and available recovery.

The core model also keeps stranding as a terminal state.

This rule does not represent rescue during an episode.

Dangerous crowding or capacity loss can occur without any skier becoming stranded.

The dissertation must therefore describe its result as detection before simulated stranding.

It must not describe the result as detection before every possible harm.

## Future work

I defer a broader harm model until the core experiment is complete.

The [broader harm extension](scope-creep/broader-harm-model.md) records that possible work.
